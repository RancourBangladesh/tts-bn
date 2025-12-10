#!/usr/bin/env python3
"""
Training Data Preparation for Bengali TTS

Prepares the collected recordings for TTS training:
- Creates metadata.csv in LJSpeech format
- Splits data into train/val/test
- Generates phoneme sequences using G2P
"""

import os
import csv
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

from g2p import g2p


def load_metadata(recordings_dir: str) -> List[Dict]:
    """Load all metadata from recordings directory."""
    metadata = []
    
    # Try JSONL file first
    jsonl_path = os.path.join(recordings_dir, 'metadata.jsonl')
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    metadata.append(json.loads(line))
    
    # Also check individual JSON files
    for filename in os.listdir(recordings_dir):
        if filename.endswith('.json') and filename != 'metadata.json':
            filepath = os.path.join(recordings_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data not in metadata:
                    metadata.append(data)
    
    return metadata


def create_ljspeech_metadata(metadata: List[Dict], 
                              wav_dir: str,
                              output_path: str,
                              use_phonemes: bool = True) -> int:
    """
    Create metadata.csv in LJSpeech format.
    
    Format: wav_filename|transcription|normalized_text
    
    Args:
        metadata: List of recording metadata dictionaries
        wav_dir: Directory containing WAV files
        output_path: Path to save metadata.csv
        use_phonemes: Whether to include phoneme sequences
    
    Returns:
        Number of entries written
    """
    entries = []
    
    for item in metadata:
        wav_filename = item.get('filename', '')
        text = item.get('text', '')
        
        if not wav_filename or not text:
            continue
        
        # Check if WAV file exists
        wav_path = os.path.join(wav_dir, wav_filename)
        if not os.path.exists(wav_path):
            # Try without extension variations
            base = Path(wav_filename).stem
            for ext in ['.wav', '_temp.webm']:
                alt_path = os.path.join(wav_dir, base + ext)
                if os.path.exists(alt_path):
                    wav_filename = base + ext
                    break
            else:
                continue
        
        # Get phoneme sequence
        if use_phonemes:
            phonemes = g2p(text)
        else:
            phonemes = text
        
        entries.append({
            'wav': Path(wav_filename).stem,  # LJSpeech uses stem without extension
            'text': text,
            'phonemes': phonemes
        })
    
    # Write metadata.csv
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        for entry in entries:
            line = f"{entry['wav']}|{entry['text']}|{entry['phonemes']}\n"
            f.write(line)
    
    print(f"Created {output_path} with {len(entries)} entries")
    return len(entries)


def split_dataset(metadata_path: str, 
                  output_dir: str,
                  train_ratio: float = 0.9,
                  val_ratio: float = 0.05,
                  test_ratio: float = 0.05,
                  seed: int = 42) -> Dict[str, int]:
    """
    Split metadata into train/val/test sets.
    
    Args:
        metadata_path: Path to metadata.csv
        output_dir: Directory to save split files
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with counts for each split
    """
    # Read all entries
    entries = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(line.strip())
    
    # Shuffle
    random.seed(seed)
    random.shuffle(entries)
    
    # Calculate split sizes
    total = len(entries)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_entries = entries[:train_size]
    val_entries = entries[train_size:train_size + val_size]
    test_entries = entries[train_size + val_size:]
    
    # Write split files
    os.makedirs(output_dir, exist_ok=True)
    
    splits = {
        'train': train_entries,
        'val': val_entries,
        'test': test_entries
    }
    
    counts = {}
    for split_name, split_entries in splits.items():
        output_path = os.path.join(output_dir, f'{split_name}.csv')
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in split_entries:
                f.write(entry + '\n')
        counts[split_name] = len(split_entries)
        print(f"Created {output_path} with {len(split_entries)} entries")
    
    return counts


def generate_filelist(metadata_path: str, wav_dir: str, output_path: str):
    """
    Generate filelist.txt for training (full paths).
    
    Format: /full/path/to/audio.wav|text|phonemes
    """
    entries = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    wav_name = parts[0]
                    wav_path = os.path.join(wav_dir, wav_name + '.wav')
                    if os.path.exists(wav_path):
                        wav_path = os.path.abspath(wav_path)
                        entries.append(f"{wav_path}|{'|'.join(parts[1:])}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(entry + '\n')
    
    print(f"Created {output_path} with {len(entries)} entries")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare training data for Bengali TTS')
    parser.add_argument('--recordings', type=str, default='dataset/recordings',
                        help='Directory containing recordings')
    parser.add_argument('--processed', type=str, default='dataset/processed/wav_16k_mono',
                        help='Directory containing processed WAV files')
    parser.add_argument('--output', type=str, default='dataset',
                        help='Output directory for metadata files')
    parser.add_argument('--no-phonemes', action='store_true',
                        help='Do not include phoneme sequences')
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    recordings_dir = os.path.join(base_dir, args.recordings)
    processed_dir = os.path.join(base_dir, args.processed)
    output_dir = os.path.join(base_dir, args.output)
    
    print("\n" + "=" * 60)
    print("Bengali TTS Training Data Preparation")
    print("=" * 60)
    
    # Load metadata
    print(f"\nLoading metadata from: {recordings_dir}")
    metadata = load_metadata(recordings_dir)
    print(f"Found {len(metadata)} recordings")
    
    if len(metadata) == 0:
        print("\n⚠️  No recordings found! Record some audio first.")
        return
    
    # Determine WAV directory
    wav_dir = processed_dir if os.path.exists(processed_dir) and os.listdir(processed_dir) else recordings_dir
    print(f"Using WAV files from: {wav_dir}")
    
    # Create metadata.csv
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    count = create_ljspeech_metadata(
        metadata, wav_dir, metadata_path,
        use_phonemes=not args.no_phonemes
    )
    
    if count == 0:
        print("\n⚠️  No valid entries found!")
        return
    
    # Split dataset
    print("\nSplitting dataset...")
    counts = split_dataset(metadata_path, output_dir)
    
    # Generate filelists
    print("\nGenerating filelists...")
    for split in ['train', 'val', 'test']:
        split_csv = os.path.join(output_dir, f'{split}.csv')
        split_txt = os.path.join(output_dir, f'{split}_filelist.txt')
        if os.path.exists(split_csv):
            generate_filelist(split_csv, wav_dir, split_txt)
    
    print("\n" + "=" * 60)
    print("Preparation complete!")
    print("=" * 60)
    print(f"\nTotal recordings: {count}")
    print(f"Train: {counts.get('train', 0)}")
    print(f"Val: {counts.get('val', 0)}")
    print(f"Test: {counts.get('test', 0)}")
    print(f"\nOutput directory: {output_dir}")
    print("\nNext steps:")
    print("1. Review metadata.csv for accuracy")
    print("2. Configure your TTS training framework")
    print("3. Start training!")


if __name__ == '__main__':
    main()
