#!/usr/bin/env python3
"""
Bengali TTS Trainer

Training pipeline with:
- Learning rate warmup + cosine decay
- Gradient accumulation
- Automatic Mixed Precision (AMP)
- Checkpointing with auto-resume and rollback
- Alignment monitoring and collapse detection
- Early stopping
- Augmentation scheduling
"""

import os
import json
import math
import random
import logging
import shutil
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from datetime import datetime
from collections import deque

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

from training_config import TrainingConfig, get_default_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WarmupCosineScheduler(LambdaLR):
    """Learning rate scheduler with warmup and cosine decay."""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 min_lr_ratio: float = 0.0, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, self.lr_lambda, last_epoch)
    
    def lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(self.min_lr_ratio, cosine_decay)


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001,
                 mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        if self.mode == 'min':
            is_improvement = value < (self.best_value - self.min_delta)
        else:
            is_improvement = value > (self.best_value + self.min_delta)
        
        if is_improvement:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class AlignmentMonitor:
    """Monitor training alignment to detect collapse."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.alignment_history: deque = deque(maxlen=100)
        self.mel_loss_history: deque = deque(maxlen=100)
        
    def compute_diagonal_score(self, attention: torch.Tensor) -> float:
        """
        Compute how diagonal the attention matrix is.
        Higher score = more diagonal = better alignment.
        """
        if attention is None:
            return 0.0
        
        # attention shape: [batch, decoder_steps, encoder_steps]
        if len(attention.shape) == 4:
            attention = attention.squeeze(1)  # Remove head dimension if present
        
        batch_scores = []
        for att in attention:
            # Get dimensions
            dec_len, enc_len = att.shape
            
            # Create ideal diagonal mask
            ideal = torch.zeros_like(att)
            for i in range(dec_len):
                j = int(i * enc_len / dec_len)
                if j < enc_len:
                    ideal[i, j] = 1.0
            
            # Compute overlap with diagonal
            score = (att * ideal).sum() / (att.sum() + 1e-8)
            batch_scores.append(score.item())
        
        return sum(batch_scores) / len(batch_scores) if batch_scores else 0.0
    
    def check_collapse(self, attention: torch.Tensor) -> Tuple[bool, float]:
        """Check if attention has collapsed."""
        score = self.compute_diagonal_score(attention)
        self.alignment_history.append(score)
        
        # Check for collapse
        is_collapsed = score < self.config.alignment.collapse_threshold
        
        # Also check if score is degrading
        if len(self.alignment_history) >= 10:
            recent_avg = sum(list(self.alignment_history)[-10:]) / 10
            older_avg = sum(list(self.alignment_history)[-20:-10]) / 10 if len(self.alignment_history) >= 20 else recent_avg
            is_degrading = recent_avg < older_avg * 0.8
            is_collapsed = is_collapsed or is_degrading
        
        return is_collapsed, score
    
    def log_attention_plot(self, attention: torch.Tensor, step: int, output_dir: str):
        """Save attention plot for debugging."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            if len(attention.shape) == 4:
                attention = attention[0, 0]  # First sample, first head
            elif len(attention.shape) == 3:
                attention = attention[0]  # First sample
            
            att_np = attention.detach().cpu().numpy()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(att_np, aspect='auto', origin='lower', cmap='viridis')
            ax.set_xlabel('Encoder Steps')
            ax.set_ylabel('Decoder Steps')
            ax.set_title(f'Attention Alignment - Step {step}')
            plt.colorbar(im, ax=ax)
            
            plot_dir = os.path.join(output_dir, 'attention_plots')
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f'attention_step_{step:08d}.png'), dpi=100)
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib not installed, skipping attention plot")
        except Exception as e:
            logger.warning(f"Failed to save attention plot: {e}")
    
    def log_mel_spectrogram(self, mel: torch.Tensor, step: int, output_dir: str,
                           name: str = "mel"):
        """Save mel spectrogram preview."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            if len(mel.shape) == 3:
                mel = mel[0]  # First sample
            
            mel_np = mel.detach().cpu().numpy()
            
            fig, ax = plt.subplots(figsize=(12, 4))
            im = ax.imshow(mel_np, aspect='auto', origin='lower', cmap='magma')
            ax.set_xlabel('Time')
            ax.set_ylabel('Mel Channels')
            ax.set_title(f'Mel Spectrogram - Step {step}')
            plt.colorbar(im, ax=ax)
            
            plot_dir = os.path.join(output_dir, 'mel_plots')
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f'{name}_step_{step:08d}.png'), dpi=100)
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib not installed, skipping mel plot")
        except Exception as e:
            logger.warning(f"Failed to save mel plot: {e}")


class CheckpointManager:
    """Manage checkpoints with rollback support."""
    
    def __init__(self, config: TrainingConfig, output_dir: str):
        self.config = config
        self.checkpoint_dir = os.path.join(output_dir, config.checkpoint.checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.checkpoint_history: List[Dict[str, Any]] = []
        self.best_val_loss = float('inf')
    
    def save(self, state: Dict[str, Any], step: int, val_loss: float,
             alignment_score: float, is_best: bool = False):
        """Save checkpoint."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'checkpoint_step_{step:08d}.pt'
        )
        
        # Add metadata
        state['step'] = step
        state['val_loss'] = val_loss
        state['alignment_score'] = alignment_score
        state['timestamp'] = datetime.now().isoformat()
        
        torch.save(state, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Track checkpoint
        self.checkpoint_history.append({
            'path': checkpoint_path,
            'step': step,
            'val_loss': val_loss,
            'alignment_score': alignment_score
        })
        
        # Save best checkpoint
        if is_best and self.config.checkpoint.save_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pt')
            torch.save(state, best_path)
            logger.info(f"Saved best checkpoint: {best_path}")
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints."""
        max_keep = self.config.checkpoint.max_checkpoints
        if len(self.checkpoint_history) > max_keep:
            # Keep best and most recent
            to_remove = self.checkpoint_history[:-max_keep]
            for ckpt in to_remove:
                if os.path.exists(ckpt['path']) and 'best' not in ckpt['path']:
                    os.remove(ckpt['path'])
            self.checkpoint_history = self.checkpoint_history[-max_keep:]
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        if not self.checkpoint_history:
            # Check for existing checkpoints
            checkpoints = sorted(Path(self.checkpoint_dir).glob('checkpoint_step_*.pt'))
            if checkpoints:
                return str(checkpoints[-1])
            return None
        return self.checkpoint_history[-1]['path']
    
    def get_rollback_checkpoint(self, current_alignment: float) -> Optional[str]:
        """Get checkpoint to rollback to if alignment degraded."""
        if not self.config.checkpoint.enable_rollback:
            return None
        
        threshold = self.config.checkpoint.rollback_threshold
        history_size = self.config.checkpoint.rollback_history
        
        # Check recent checkpoints for better alignment
        recent = self.checkpoint_history[-history_size:] if len(self.checkpoint_history) >= history_size else self.checkpoint_history
        
        for ckpt in reversed(recent):
            if ckpt['alignment_score'] > current_alignment / threshold:
                logger.warning(f"Alignment degraded. Rolling back to step {ckpt['step']}")
                return ckpt['path']
        
        return None
    
    def load(self, path: str) -> Dict[str, Any]:
        """Load checkpoint."""
        return torch.load(path, map_location='cpu')


class DataAugmentor:
    """Data augmentation with automatic disabling."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.aug_config = config.augmentation
    
    def augment(self, audio: torch.Tensor, current_epoch: int, 
                total_epochs: int) -> torch.Tensor:
        """Apply augmentation if within training window."""
        if not self.aug_config.should_augment(current_epoch, total_epochs):
            return audio
        
        # Speed perturbation
        if self.aug_config.speed_perturbation:
            factor = random.choice(self.aug_config.speed_factors)
            if factor != 1.0:
                audio = self._speed_perturb(audio, factor)
        
        # Volume scaling
        if self.aug_config.volume_scaling:
            scale = random.uniform(*self.aug_config.volume_range)
            audio = audio * scale
        
        return audio
    
    def _speed_perturb(self, audio: torch.Tensor, factor: float) -> torch.Tensor:
        """Apply speed perturbation."""
        try:
            import torchaudio.transforms as T
            # Resample to change speed
            orig_freq = self.config.audio.training_sample_rate
            new_freq = int(orig_freq * factor)
            resampler = T.Resample(orig_freq, new_freq)
            audio = resampler(audio)
            # Resample back to original rate
            resampler_back = T.Resample(new_freq, orig_freq)
            audio = resampler_back(audio)
        except Exception as e:
            logger.warning(f"Speed perturbation failed: {e}")
        return audio


class BengaliTTSTrainer:
    """Main trainer class for Bengali TTS."""
    
    def __init__(self, config: Optional[TrainingConfig] = None,
                 model: Optional[nn.Module] = None,
                 optimizer_g: Optional[torch.optim.Optimizer] = None,
                 optimizer_d: Optional[torch.optim.Optimizer] = None):
        
        self.config = config or get_default_config()
        self.model = model
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        
        # Output directory
        self.output_dir = self.config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.checkpoint_manager = CheckpointManager(self.config, self.output_dir)
        self.alignment_monitor = AlignmentMonitor(self.config)
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping.patience,
            min_delta=self.config.early_stopping.min_delta,
            mode='min'
        )
        self.augmentor = DataAugmentor(self.config)
        
        # AMP scaler
        self.scaler = GradScaler(enabled=self.config.amp.enabled)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Gradient accumulation counter
        self.accumulation_counter = 0
        
        # Metrics history
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.alignment_scores: List[float] = []
    
    def setup_schedulers(self, total_steps: int):
        """Setup learning rate schedulers."""
        self.scheduler_g = WarmupCosineScheduler(
            self.optimizer_g,
            warmup_steps=self.config.optimizer.warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=self.config.optimizer.min_lr / self.config.optimizer.generator_lr
        )
        
        if self.optimizer_d is not None:
            self.scheduler_d = WarmupCosineScheduler(
                self.optimizer_d,
                warmup_steps=self.config.optimizer.warmup_steps,
                total_steps=total_steps,
                min_lr_ratio=self.config.optimizer.min_lr / self.config.optimizer.discriminator_lr
            )
    
    def train_step(self, batch: Dict[str, torch.Tensor], 
                   epoch: int) -> Dict[str, float]:
        """
        Single training step with gradient accumulation and AMP.
        
        Returns dictionary of losses.
        """
        self.model.train()
        
        # Apply augmentation
        if 'audio' in batch:
            batch['audio'] = self.augmentor.augment(
                batch['audio'], 
                epoch, 
                self.config.epochs
            )
        
        # Forward pass with AMP
        with autocast(enabled=self.config.amp.enabled):
            outputs = self.model(batch)
            loss = outputs.get('loss', outputs.get('mel_loss', torch.tensor(0.0)))
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.batching.gradient_accumulation_steps
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        self.accumulation_counter += 1
        
        # Optimizer step after accumulation
        if self.accumulation_counter >= self.config.batching.gradient_accumulation_steps:
            # Gradient clipping
            if self.config.optimizer.grad_clip_thresh > 0:
                self.scaler.unscale_(self.optimizer_g)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.optimizer.grad_clip_thresh
                )
            
            # Optimizer step
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
            
            # Zero gradients
            self.optimizer_g.zero_grad()
            
            # Reset counter
            self.accumulation_counter = 0
            
            # Update learning rate
            self.scheduler_g.step()
            
            self.global_step += 1
        
        # Get attention for monitoring
        attention = outputs.get('attention', outputs.get('attn', None))
        
        return {
            'loss': loss.item() * self.config.batching.gradient_accumulation_steps,
            'attention': attention,
            'mel_output': outputs.get('mel_output', outputs.get('mel', None))
        }
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """Run validation and return loss and alignment score."""
        self.model.eval()
        
        total_loss = 0.0
        total_alignment = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                with autocast(enabled=self.config.amp.enabled):
                    outputs = self.model(batch)
                    loss = outputs.get('loss', outputs.get('mel_loss', torch.tensor(0.0)))
                
                total_loss += loss.item()
                
                # Check alignment
                attention = outputs.get('attention', outputs.get('attn', None))
                if attention is not None:
                    alignment_score = self.alignment_monitor.compute_diagonal_score(attention)
                    total_alignment += alignment_score
                
                num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        avg_alignment = total_alignment / max(1, num_batches)
        
        return avg_loss, avg_alignment
    
    def should_stop(self, val_loss: float, alignment_score: float) -> bool:
        """Check if training should stop."""
        # Early stopping on validation loss
        if self.config.early_stopping.enabled:
            if self.early_stopping(val_loss):
                logger.info("Early stopping triggered: validation loss not improving")
                return True
        
        # Stop on alignment degradation
        if self.config.early_stopping.stop_on_alignment_degradation:
            if alignment_score < self.config.early_stopping.alignment_threshold:
                logger.warning(f"Stopping: alignment score ({alignment_score:.4f}) below threshold")
                return True
        
        return False
    
    def save_checkpoint(self, val_loss: float, alignment_score: float):
        """Save training checkpoint."""
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
        
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict()
        }
        
        if self.optimizer_d is not None:
            state['optimizer_d_state_dict'] = self.optimizer_d.state_dict()
            state['scheduler_d_state_dict'] = self.scheduler_d.state_dict()
        
        self.checkpoint_manager.save(
            state, self.global_step, val_loss, alignment_score, is_best
        )
    
    def load_checkpoint(self, path: Optional[str] = None):
        """Load checkpoint for resuming training."""
        if path is None and self.config.checkpoint.auto_resume:
            path = self.checkpoint_manager.get_latest_checkpoint()
        
        if path is None or not os.path.exists(path):
            logger.info("No checkpoint found, starting from scratch")
            return
        
        logger.info(f"Loading checkpoint: {path}")
        checkpoint = self.checkpoint_manager.load(path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        
        if 'scheduler_g_state_dict' in checkpoint:
            self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.optimizer_d is not None and 'optimizer_d_state_dict' in checkpoint:
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        
        logger.info(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
    
    def log_metrics(self, step: int, train_loss: float, attention: torch.Tensor = None,
                    mel_output: torch.Tensor = None):
        """Log training metrics."""
        self.train_losses.append(train_loss)
        
        # Log to console
        if step % self.config.log_steps == 0:
            lr = self.scheduler_g.get_last_lr()[0]
            logger.info(f"Step {step}: loss={train_loss:.4f}, lr={lr:.2e}")
        
        # Log attention plot
        if attention is not None and step % self.config.alignment.log_attention_steps == 0:
            self.alignment_monitor.log_attention_plot(attention, step, self.output_dir)
        
        # Log mel spectrogram
        if mel_output is not None and step % self.config.alignment.log_mel_steps == 0:
            self.alignment_monitor.log_mel_spectrogram(mel_output, step, self.output_dir)
    
    def check_alignment_collapse(self, attention: torch.Tensor) -> bool:
        """Check if alignment has collapsed."""
        if attention is None or not self.config.alignment.stop_on_collapse:
            return False
        
        is_collapsed, score = self.alignment_monitor.check_collapse(attention)
        
        if is_collapsed:
            logger.warning(f"Alignment collapse detected! Score: {score:.4f}")
            
            # Try rollback
            rollback_path = self.checkpoint_manager.get_rollback_checkpoint(score)
            if rollback_path:
                self.load_checkpoint(rollback_path)
                return False  # Continue training from rollback
            
            return True  # Stop training
        
        return False


def create_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Create optimizer with configured learning rate."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.generator_lr,
        betas=config.optimizer.betas,
        weight_decay=config.optimizer.weight_decay
    )


if __name__ == '__main__':
    # Test configuration and trainer setup
    config = get_default_config()
    print("Trainer module loaded successfully")
    print(f"Config: batch_size={config.batching.batch_size}, "
          f"accumulation={config.batching.gradient_accumulation_steps}, "
          f"effective_batch={config.batching.batch_size * config.batching.gradient_accumulation_steps}")
    print(f"AMP enabled: {config.amp.enabled}")
    print(f"Warmup steps: {config.optimizer.warmup_steps}")
