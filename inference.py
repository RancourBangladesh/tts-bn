#!/usr/bin/env python3
"""
Bengali TTS Inference CLI

Command-line interface for text-to-speech synthesis.
Supports multiple output formats and model backends.

Usage:
    python inference.py --text "আমার সোনার বাংলা" --output output.wav
    python inference.py --text "Hello" --output hello.wav --model checkpoints/best_checkpoint.pt
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from g2p import g2p, normalize_text


def load_model(model_path: str, device: str = 'cuda'):
    """
    Load TTS model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        Loaded model
    """
    import torch
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model configuration
    config = checkpoint.get('config', {})
    
    # Initialize model based on config
    model_type = config.get('model', {}).get('model_type', 'vits')
    
    print(f"Loading {model_type} model from {model_path}")
    
    # TODO: Initialize actual model architecture here
    # This is a placeholder that should be replaced with actual model loading
    # For now, we'll return the checkpoint itself
    
    return checkpoint, config, device


def synthesize(text: str, model, config: Dict, device: str,
               speaker_id: int = 0) -> Optional[Any]:
    """
    Synthesize speech from text.
    
    Args:
        text: Input text (Bengali or mixed)
        model: Loaded TTS model
        config: Model configuration
        device: Device for inference
        speaker_id: Speaker ID for multi-speaker models
    
    Returns:
        Audio waveform as numpy array
    """
    import torch
    
    # Normalize and convert to phonemes
    phonemes = g2p(text)
    print(f"Phonemes: {phonemes}")
    
    # TODO: Implement actual synthesis
    # This requires the full model architecture
    # For now, we return None to indicate this is a placeholder
    
    print("Note: Full synthesis not yet implemented. Model training required first.")
    return None


def save_audio(waveform, output_path: str, sample_rate: int = 22050):
    """
    Save audio waveform to file.
    
    Args:
        waveform: Audio data as numpy array
        output_path: Output file path
        sample_rate: Sample rate in Hz
    """
    import numpy as np
    
    if waveform is None:
        print("No waveform to save")
        return False
    
    try:
        import soundfile as sf
        sf.write(output_path, waveform, sample_rate)
        print(f"Saved audio to: {output_path}")
        return True
    except ImportError:
        # Fallback to scipy
        try:
            from scipy.io import wavfile
            # Normalize to int16
            if waveform.dtype != np.int16:
                waveform = (waveform * 32767).astype(np.int16)
            wavfile.write(output_path, sample_rate, waveform)
            print(f"Saved audio to: {output_path}")
            return True
        except ImportError:
            print("Error: Neither soundfile nor scipy available for audio export")
            return False


def export_onnx(model, output_path: str, config: Dict):
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model
        output_path: Path for ONNX file
        config: Model configuration
    """
    import torch
    
    print(f"Exporting model to ONNX: {output_path}")
    
    # TODO: Implement ONNX export
    # This requires the actual model architecture
    print("ONNX export not yet implemented - requires trained model")


def export_torchscript(model, output_path: str, config: Dict):
    """
    Export model to TorchScript format.
    
    Args:
        model: PyTorch model
        output_path: Path for TorchScript file
        config: Model configuration
    """
    import torch
    
    print(f"Exporting model to TorchScript: {output_path}")
    
    # TODO: Implement TorchScript export
    # This requires the actual model architecture
    print("TorchScript export not yet implemented - requires trained model")


def main():
    parser = argparse.ArgumentParser(
        description='Bengali TTS Inference CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic synthesis
  python inference.py --text "আমার সোনার বাংলা" --output output.wav
  
  # With specific model
  python inference.py --text "Hello" --output hello.wav --model checkpoints/best.pt
  
  # Just convert to phonemes (no synthesis)
  python inference.py --text "বাংলাদেশ" --phonemes-only
  
  # Export model to ONNX
  python inference.py --model checkpoints/best.pt --export-onnx model.onnx
'''
    )
    
    # Input options
    parser.add_argument('--text', '-t', type=str, help='Text to synthesize')
    parser.add_argument('--file', '-f', type=str, help='File containing text to synthesize')
    
    # Output options
    parser.add_argument('--output', '-o', type=str, default='output.wav',
                        help='Output audio file path')
    parser.add_argument('--sample-rate', type=int, default=22050,
                        help='Output sample rate')
    
    # Model options
    parser.add_argument('--model', '-m', type=str, default='checkpoints/best_checkpoint.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device for inference')
    parser.add_argument('--speaker', type=int, default=0,
                        help='Speaker ID for multi-speaker models')
    
    # G2P options
    parser.add_argument('--phonemes-only', action='store_true',
                        help='Only output phonemes, no synthesis')
    parser.add_argument('--no-number-expansion', action='store_true',
                        help='Do not expand numbers to words')
    parser.add_argument('--no-english-transliteration', action='store_true',
                        help='Do not transliterate English words')
    
    # Export options
    parser.add_argument('--export-onnx', type=str, metavar='PATH',
                        help='Export model to ONNX format')
    parser.add_argument('--export-torchscript', type=str, metavar='PATH',
                        help='Export model to TorchScript format')
    
    # Other options
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.text and not args.file and not args.export_onnx and not args.export_torchscript:
        parser.error('Either --text, --file, --export-onnx, or --export-torchscript is required')
    
    # Get text input
    text = args.text
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    
    # Phonemes-only mode
    if args.phonemes_only and text:
        normalized = normalize_text(
            text,
            expand_nums=not args.no_number_expansion,
            transliterate_eng=not args.no_english_transliteration
        )
        phonemes = g2p(
            text,
            expand_nums=not args.no_number_expansion,
            transliterate_eng=not args.no_english_transliteration
        )
        
        print(f"Input:      {text}")
        print(f"Normalized: {normalized}")
        print(f"Phonemes:   {phonemes}")
        return
    
    # Model export mode
    if args.export_onnx or args.export_torchscript:
        try:
            model, config, device = load_model(args.model, args.device)
            
            if args.export_onnx:
                export_onnx(model, args.export_onnx, config)
            
            if args.export_torchscript:
                export_torchscript(model, args.export_torchscript, config)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        return
    
    # Synthesis mode
    if text:
        print(f"\n{'='*60}")
        print("Bengali TTS Inference")
        print('='*60)
        print(f"Input text: {text}")
        
        # Check if model exists
        if not os.path.exists(args.model):
            print(f"\nWarning: Model not found at {args.model}")
            print("Running in phoneme-only mode...\n")
            
            phonemes = g2p(
                text,
                expand_nums=not args.no_number_expansion,
                transliterate_eng=not args.no_english_transliteration
            )
            print(f"Phonemes: {phonemes}")
            print("\nTo synthesize audio, train a model first using:")
            print("  python train.py --config training_config.json")
            return
        
        try:
            model, config, device = load_model(args.model, args.device)
            
            # Synthesize
            waveform = synthesize(
                text, model, config, device,
                speaker_id=args.speaker
            )
            
            if waveform is not None:
                # Save audio
                save_audio(waveform, args.output, args.sample_rate)
                print(f"\nOutput saved to: {args.output}")
            else:
                print("\nSynthesis not yet implemented - model training required first")
                
        except Exception as e:
            print(f"Error during synthesis: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()
