#!/usr/bin/env python3
"""
Training Configuration for Bengali TTS

Comprehensive training settings optimized for:
- Training stability (prevents collapse)
- GTX 1660 Super GPU (6GB VRAM)
- Natural Bangladeshi Bengali voice
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class OptimizerConfig:
    """Optimizer and learning rate configuration."""
    
    # Learning rates (reduced 3-5x for stability)
    generator_lr: float = 1e-4  # Reduced from 2e-4
    discriminator_lr: float = 5e-5  # Reduced from 2e-4
    
    # Betas for Adam optimizer
    betas: tuple = (0.8, 0.99)
    
    # Weight decay
    weight_decay: float = 1e-6
    
    # Gradient clipping
    grad_clip_thresh: float = 1.0
    
    # Learning rate scheduler
    scheduler_type: str = "warmup_cosine"  # Options: warmup_cosine, step, exponential
    
    # Warmup settings
    warmup_steps: int = 4000
    warmup_start_lr: float = 1e-7
    
    # Cosine decay settings
    min_lr: float = 1e-6
    
    # Step decay settings (if using step scheduler)
    lr_decay_step: int = 10000
    lr_decay_factor: float = 0.5


@dataclass
class BatchingConfig:
    """Batching configuration for GPU stability (GTX 1660 Super 6GB)."""
    
    # Batch size (reduced to fit in 6GB VRAM)
    batch_size: int = 8  # Reduced from 16-32
    
    # Gradient accumulation steps (effective batch size = batch_size * accumulation_steps)
    gradient_accumulation_steps: int = 4  # Effective batch = 32
    
    # Number of workers for data loading
    num_workers: int = 4
    
    # Pin memory for faster GPU transfer
    pin_memory: bool = True
    
    # Drop last incomplete batch
    drop_last: bool = True


@dataclass
class AMPConfig:
    """Automatic Mixed Precision configuration for memory efficiency."""
    
    # Enable AMP
    enabled: bool = True
    
    # Initial loss scale
    init_scale: float = 2.0 ** 16
    
    # Scale growth factor
    growth_factor: float = 2.0
    
    # Scale backoff factor
    backoff_factor: float = 0.5
    
    # Growth interval (steps)
    growth_interval: int = 2000
    
    # Enable dynamic loss scaling
    dynamic_loss_scaling: bool = True


@dataclass
class CheckpointConfig:
    """Checkpointing configuration with rollback support."""
    
    # Save checkpoint every N steps
    save_steps: int = 1000
    
    # Maximum number of checkpoints to keep
    max_checkpoints: int = 10
    
    # Save best checkpoint based on validation loss
    save_best: bool = True
    
    # Enable auto-resume from last checkpoint
    auto_resume: bool = True
    
    # Checkpoint directory
    checkpoint_dir: str = "checkpoints"
    
    # Keep track of checkpoint quality for rollback
    enable_rollback: bool = True
    
    # Rollback threshold (if alignment score drops below this %, rollback)
    rollback_threshold: float = 0.7
    
    # Number of previous checkpoints to consider for rollback
    rollback_history: int = 3


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration to prevent overfitting/collapse."""
    
    # Enable early stopping
    enabled: bool = True
    
    # Patience (number of validation checks without improvement)
    patience: int = 10
    
    # Minimum delta for improvement
    min_delta: float = 0.001
    
    # Monitor metric
    monitor: str = "val_loss"  # Options: val_loss, alignment_score, mel_loss
    
    # Stop if alignment score degrades
    stop_on_alignment_degradation: bool = True
    
    # Alignment degradation threshold
    alignment_threshold: float = 0.5


@dataclass
class AlignmentMonitorConfig:
    """Alignment monitoring for detecting training collapse."""
    
    # Enable alignment monitoring
    enabled: bool = True
    
    # Log attention plots every N steps
    log_attention_steps: int = 500
    
    # Log mel spectrogram previews every N steps
    log_mel_steps: int = 500
    
    # Save audio samples every N steps
    save_audio_steps: int = 1000
    
    # Number of samples to generate
    num_samples: int = 3
    
    # Stop training if alignment becomes fuzzy/collapsed
    stop_on_collapse: bool = True
    
    # Collapse detection threshold (diagonal attention ratio)
    collapse_threshold: float = 0.3


@dataclass
class AugmentationConfig:
    """Data augmentation with automatic late-training disabling."""
    
    # Enable augmentation scheduling
    enabled: bool = True
    
    # Augmentation cutoff (disable after this % of training)
    cutoff_epoch_ratio: float = 0.35  # Disable after 35% of epochs
    
    # Speed perturbation
    speed_perturbation: bool = True
    speed_factors: List[float] = field(default_factory=lambda: [0.95, 1.0, 1.05])
    
    # Volume scaling
    volume_scaling: bool = True
    volume_range: tuple = (0.9, 1.1)
    
    # Noise injection
    noise_injection: bool = True
    noise_snr_range: tuple = (30, 50)  # dB
    
    # Time stretching (DISABLED by default - can cause artifacts)
    time_stretch: bool = False
    time_stretch_range: tuple = (0.95, 1.05)
    
    def should_augment(self, current_epoch: int, total_epochs: int) -> bool:
        """Check if augmentation should be applied based on training progress."""
        if not self.enabled:
            return False
        progress = current_epoch / total_epochs
        return progress < self.cutoff_epoch_ratio


@dataclass  
class AudioConfig:
    """Audio processing configuration."""
    
    # Sample rate (48kHz for high quality recording, 22050 for training)
    recording_sample_rate: int = 48000
    training_sample_rate: int = 22050
    
    # Channels
    channels: int = 1  # Mono
    
    # Bit depth
    bit_depth: int = 16
    
    # Mel spectrogram settings
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0
    
    # Normalization
    target_lufs: float = -14.0  # Loudness target
    
    # Quality thresholds
    min_duration: float = 0.5  # seconds
    max_duration: float = 15.0  # seconds
    max_clipping_ratio: float = 0.01
    min_snr: float = 20.0  # dB


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Model type
    model_type: str = "vits"  # Options: tacotron2, vits, fastspeech2
    
    # Hidden dimensions
    hidden_dim: int = 192
    
    # Encoder settings
    encoder_layers: int = 6
    encoder_heads: int = 2
    
    # Decoder settings
    decoder_layers: int = 6
    decoder_heads: int = 2
    
    # Flow settings (for VITS)
    flow_hidden_dim: int = 192
    flow_layers: int = 4
    
    # Posterior encoder (for VITS)
    posterior_encoder_dim: int = 192
    
    # Vocabulary size (Bengali phonemes + special tokens)
    vocab_size: int = 200


@dataclass
class TrainingConfig:
    """Main training configuration combining all sub-configs."""
    
    # Dataset paths
    dataset_dir: str = "dataset"
    train_file: str = "train_filelist.txt"
    val_file: str = "val_filelist.txt"
    
    # Output directory
    output_dir: str = "output"
    
    # Training epochs
    epochs: int = 1000
    
    # Validation frequency (validate every N steps)
    val_steps: int = 1000
    
    # Logging frequency
    log_steps: int = 100
    
    # Random seed
    seed: int = 42
    
    # Device
    device: str = "cuda"  # or "cpu"
    
    # Sub-configurations
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    amp: AMPConfig = field(default_factory=AMPConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    alignment: AlignmentMonitorConfig = field(default_factory=AlignmentMonitorConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        import dataclasses
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        # Handle nested dataclasses
        if 'optimizer' in d and isinstance(d['optimizer'], dict):
            d['optimizer'] = OptimizerConfig(**d['optimizer'])
        if 'batching' in d and isinstance(d['batching'], dict):
            d['batching'] = BatchingConfig(**d['batching'])
        if 'amp' in d and isinstance(d['amp'], dict):
            d['amp'] = AMPConfig(**d['amp'])
        if 'checkpoint' in d and isinstance(d['checkpoint'], dict):
            d['checkpoint'] = CheckpointConfig(**d['checkpoint'])
        if 'early_stopping' in d and isinstance(d['early_stopping'], dict):
            d['early_stopping'] = EarlyStoppingConfig(**d['early_stopping'])
        if 'alignment' in d and isinstance(d['alignment'], dict):
            d['alignment'] = AlignmentMonitorConfig(**d['alignment'])
        if 'augmentation' in d and isinstance(d['augmentation'], dict):
            d['augmentation'] = AugmentationConfig(**d['augmentation'])
        if 'audio' in d and isinstance(d['audio'], dict):
            d['audio'] = AudioConfig(**d['audio'])
        if 'model' in d and isinstance(d['model'], dict):
            d['model'] = ModelConfig(**d['model'])
        return cls(**d)
    
    def save(self, path: str):
        """Save config to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load config from JSON file."""
        import json
        with open(path, 'r') as f:
            d = json.load(f)
        return cls.from_dict(d)


def get_default_config() -> TrainingConfig:
    """Get default training configuration optimized for Bengali TTS."""
    return TrainingConfig()


def get_gtx1660_config() -> TrainingConfig:
    """Get configuration optimized for GTX 1660 Super (6GB VRAM)."""
    config = TrainingConfig()
    
    # Reduce memory usage
    config.batching.batch_size = 8
    config.batching.gradient_accumulation_steps = 4
    config.amp.enabled = True
    
    # Reduce model size slightly for memory
    config.model.hidden_dim = 192
    
    return config


if __name__ == '__main__':
    # Print default configuration
    config = get_default_config()
    print("Default Training Configuration:")
    print("=" * 60)
    
    import json
    print(json.dumps(config.to_dict(), indent=2))
    
    # Save example config
    config.save('training_config.json')
    print(f"\nSaved config to training_config.json")
