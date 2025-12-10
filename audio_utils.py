#!/usr/bin/env python3
"""
Audio Processing Utilities for Bengali TTS

Functions for:
- Audio format conversion
- Silence trimming
- Loudness normalization
- Quality checks (clipping, SNR, duration)
- Batch processing
"""

import os
import argparse
import subprocess
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import wave
import struct
import math

# Try to import optional libraries
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


def get_audio_info(filepath: str) -> Optional[Dict]:
    """Get basic audio file information."""
    try:
        with wave.open(filepath, 'rb') as w:
            return {
                'channels': w.getnchannels(),
                'sample_rate': w.getframerate(),
                'bits': w.getsampwidth() * 8,
                'frames': w.getnframes(),
                'duration': w.getnframes() / w.getframerate()
            }
    except Exception as e:
        return None


def validate_path(filepath: str) -> bool:
    """
    Validate file path to prevent path traversal and command injection.
    
    Args:
        filepath: Path to validate
    
    Returns:
        True if path is safe, False otherwise
    """
    # Check for path traversal attempts
    if '..' in filepath:
        return False
    # Check for shell metacharacters
    dangerous_chars = ['|', ';', '&', '$', '`', '>', '<', '!', '\n', '\r']
    for char in dangerous_chars:
        if char in filepath:
            return False
    return True


def convert_to_wav(input_path: str, output_path: str, 
                   sample_rate: int = 16000, channels: int = 1, 
                   bits: int = 16) -> bool:
    """
    Convert audio file to WAV format using ffmpeg.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output WAV file
        sample_rate: Target sample rate (default: 16000)
        channels: Number of channels (default: 1 for mono)
        bits: Bit depth (default: 16)
    
    Returns:
        True if conversion successful, False otherwise
    """
    # Validate paths
    if not validate_path(input_path) or not validate_path(output_path):
        print(f"Security error: Invalid path characters detected")
        return False
    
    try:
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-ar', str(sample_rate),
            '-ac', str(channels),
            '-c:a', f'pcm_s{bits}le',
            output_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Conversion error: {e.stderr.decode() if e.stderr else str(e)}")
        return False
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg.")
        return False


def trim_silence(input_path: str, output_path: str,
                 threshold_db: float = -40, 
                 min_silence_duration: float = 0.1) -> bool:
    """
    Trim leading and trailing silence from audio file using ffmpeg.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output file
        threshold_db: Silence threshold in dB (default: -40)
        min_silence_duration: Minimum silence duration in seconds
    
    Returns:
        True if successful
    """
    try:
        # Use ffmpeg silenceremove filter
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-af', f'silenceremove=start_periods=1:start_duration={min_silence_duration}:'
                   f'start_threshold={threshold_db}dB:'
                   f'stop_periods=1:stop_duration={min_silence_duration}:'
                   f'stop_threshold={threshold_db}dB',
            output_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception as e:
        print(f"Trim error: {e}")
        return False


def normalize_loudness(input_path: str, output_path: str,
                       target_lufs: float = -23.0) -> bool:
    """
    Normalize audio loudness using ffmpeg loudnorm filter.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file
        target_lufs: Target loudness in LUFS (default: -23)
    
    Returns:
        True if successful
    """
    try:
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-af', f'loudnorm=I={target_lufs}:TP=-1.5:LRA=11',
            output_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception as e:
        print(f"Normalization error: {e}")
        return False


def check_clipping(filepath: str, threshold: float = 0.99) -> Tuple[bool, float]:
    """
    Check if audio file has clipping.
    
    Args:
        filepath: Path to WAV file
        threshold: Peak threshold (0.0 to 1.0)
    
    Returns:
        Tuple of (has_clipping, max_peak)
    """
    if not HAS_NUMPY:
        print("Warning: numpy not installed, skipping clipping check")
        return (False, 0.0)
    
    try:
        with wave.open(filepath, 'rb') as w:
            frames = w.readframes(w.getnframes())
            if w.getsampwidth() == 2:  # 16-bit
                samples = np.frombuffer(frames, dtype=np.int16)
                max_val = 32767
            else:  # 8-bit
                samples = np.frombuffer(frames, dtype=np.uint8)
                max_val = 255
            
            peak = np.max(np.abs(samples)) / max_val
            return (peak >= threshold, float(peak))
    except Exception as e:
        print(f"Clipping check error: {e}")
        return (False, 0.0)


def calculate_rms(filepath: str) -> Optional[float]:
    """
    Calculate RMS (Root Mean Square) of audio file.
    
    Args:
        filepath: Path to WAV file
    
    Returns:
        RMS value in dB, or None on error
    """
    if not HAS_NUMPY:
        print("Warning: numpy not installed, skipping RMS calculation")
        return None
    
    try:
        with wave.open(filepath, 'rb') as w:
            frames = w.readframes(w.getnframes())
            if w.getsampwidth() == 2:
                samples = np.frombuffer(frames, dtype=np.int16).astype(float)
                samples /= 32767.0
            else:
                samples = np.frombuffer(frames, dtype=np.uint8).astype(float)
                samples = (samples - 128) / 128.0
            
            rms = np.sqrt(np.mean(samples ** 2))
            if rms > 0:
                rms_db = 20 * math.log10(rms)
                return rms_db
            return -100.0
    except Exception as e:
        print(f"RMS calculation error: {e}")
        return None


def check_duration(filepath: str, min_dur: float = 0.2, max_dur: float = 10.0) -> Tuple[bool, float]:
    """
    Check if audio duration is within acceptable range.
    
    Args:
        filepath: Path to WAV file
        min_dur: Minimum duration in seconds
        max_dur: Maximum duration in seconds
    
    Returns:
        Tuple of (is_valid, duration)
    """
    info = get_audio_info(filepath)
    if info is None:
        return (False, 0.0)
    
    duration = info['duration']
    is_valid = min_dur <= duration <= max_dur
    return (is_valid, duration)


def quality_check(filepath: str) -> Dict:
    """
    Run all quality checks on an audio file.
    
    Args:
        filepath: Path to WAV file
    
    Returns:
        Dictionary with check results
    """
    results = {
        'filepath': filepath,
        'exists': os.path.exists(filepath),
        'checks': {}
    }
    
    if not results['exists']:
        return results
    
    # Get info
    info = get_audio_info(filepath)
    results['info'] = info
    
    # Duration check
    dur_valid, duration = check_duration(filepath)
    results['checks']['duration'] = {
        'valid': dur_valid,
        'value': duration,
        'unit': 'seconds'
    }
    
    # Clipping check
    has_clipping, peak = check_clipping(filepath)
    results['checks']['clipping'] = {
        'valid': not has_clipping,
        'value': peak,
        'unit': 'peak_ratio'
    }
    
    # RMS check
    rms = calculate_rms(filepath)
    results['checks']['rms'] = {
        'valid': rms is not None and -35 <= rms <= -10,
        'value': rms,
        'unit': 'dB'
    }
    
    # Overall
    results['passed'] = all(
        c['valid'] for c in results['checks'].values() if 'valid' in c
    )
    
    return results


def process_directory(input_dir: str, output_dir: str,
                      sample_rate: int = 16000,
                      trim: bool = True,
                      normalize: bool = True) -> List[Dict]:
    """
    Process all audio files in a directory.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        sample_rate: Target sample rate
        trim: Whether to trim silence
        normalize: Whether to normalize loudness
    
    Returns:
        List of processing results
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    extensions = {'.wav', '.mp3', '.webm', '.ogg', '.m4a', '.flac'}
    
    for filename in os.listdir(input_dir):
        ext = Path(filename).suffix.lower()
        if ext not in extensions:
            continue
        
        input_path = os.path.join(input_dir, filename)
        base_name = Path(filename).stem
        output_path = os.path.join(output_dir, f"{base_name}.wav")
        
        result = {
            'input': input_path,
            'output': output_path,
            'steps': []
        }
        
        # Convert to WAV
        temp_path = os.path.join(output_dir, f"{base_name}_temp.wav")
        if convert_to_wav(input_path, temp_path, sample_rate):
            result['steps'].append('convert')
            current_path = temp_path
        else:
            result['error'] = 'Conversion failed'
            results.append(result)
            continue
        
        # Trim silence
        if trim:
            trimmed_path = os.path.join(output_dir, f"{base_name}_trimmed.wav")
            if trim_silence(current_path, trimmed_path):
                result['steps'].append('trim')
                os.remove(current_path)
                current_path = trimmed_path
        
        # Normalize
        if normalize:
            if normalize_loudness(current_path, output_path):
                result['steps'].append('normalize')
                os.remove(current_path)
            else:
                os.rename(current_path, output_path)
        else:
            os.rename(current_path, output_path)
        
        # Quality check
        qc = quality_check(output_path)
        result['quality'] = qc
        result['success'] = True
        
        results.append(result)
        print(f"Processed: {filename} -> {Path(output_path).name}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Audio processing utilities for Bengali TTS')
    parser.add_argument('--process', action='store_true', help='Process recordings directory')
    parser.add_argument('--input', type=str, default='dataset/recordings', help='Input directory')
    parser.add_argument('--output', type=str, default='dataset/processed/wav_16k_mono', help='Output directory')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Target sample rate')
    parser.add_argument('--no-trim', action='store_true', help='Skip silence trimming')
    parser.add_argument('--no-normalize', action='store_true', help='Skip loudness normalization')
    parser.add_argument('--check', type=str, help='Run quality check on single file')
    
    args = parser.parse_args()
    
    if args.check:
        result = quality_check(args.check)
        print(json.dumps(result, indent=2))
    elif args.process:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(base_dir, args.input)
        output_dir = os.path.join(base_dir, args.output)
        
        print(f"\nProcessing audio files...")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Sample rate: {args.sample_rate}")
        print(f"Trim silence: {not args.no_trim}")
        print(f"Normalize: {not args.no_normalize}")
        print()
        
        results = process_directory(
            input_dir, output_dir,
            sample_rate=args.sample_rate,
            trim=not args.no_trim,
            normalize=not args.no_normalize
        )
        
        print(f"\nProcessed {len(results)} files")
        passed = sum(1 for r in results if r.get('success'))
        print(f"Successful: {passed}")
        print(f"Failed: {len(results) - passed}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
