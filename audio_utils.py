#!/usr/bin/env python3
"""
Audio Processing Utilities for Bengali TTS

Functions for:
- Audio format conversion (48kHz for recording, 22050Hz for training)
- Silence trimming with configurable thresholds
- Loudness normalization to -14 LUFS
- Quality checks (clipping, SNR, duration, noise detection)
- Batch processing with filtering
- Optional augmentation (speed perturbation, volume scaling)
"""

import os
import argparse
import subprocess
import json
import unicodedata
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set
import wave
import struct
import math
import hashlib
import re

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


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Audio format settings
RECORDING_SAMPLE_RATE = 48000  # High quality for recording
TRAINING_SAMPLE_RATE = 22050   # Standard for TTS training
DEFAULT_CHANNELS = 1           # Mono
DEFAULT_BIT_DEPTH = 16         # 16-bit PCM

# Loudness normalization
TARGET_LUFS = -14.0  # EBU R128 recommendation for speech

# Quality thresholds
MIN_DURATION = 0.5     # seconds
MAX_DURATION = 15.0    # seconds
CLIPPING_THRESHOLD = 0.99
MIN_RMS_DB = -35.0     # Minimum acceptable RMS
MAX_RMS_DB = -10.0     # Maximum acceptable RMS (avoid clipping)
MIN_SNR_DB = 20.0      # Minimum signal-to-noise ratio

# Silence detection
SILENCE_THRESHOLD_DB = -40.0
MIN_SILENCE_DURATION = 0.1  # seconds
TARGET_LEADING_SILENCE = 0.1  # Target silence at start
TARGET_TRAILING_SILENCE = 0.1  # Target silence at end


def get_audio_info(filepath: str) -> Optional[Dict]:
    """Get basic audio file information."""
    # Try soundfile first (handles more formats including WAVE_FORMAT_EXTENSIBLE)
    if HAS_SOUNDFILE:
        try:
            info = sf.info(filepath)
            # Extract bit depth from subtype (e.g., 'PCM_16' -> 16)
            bits = 16  # Default
            try:
                subtype = info.subtype
                if 'PCM_16' in subtype or '16' in subtype:
                    bits = 16
                elif 'PCM_24' in subtype or '24' in subtype:
                    bits = 24
                elif 'PCM_32' in subtype or '32' in subtype:
                    bits = 32
                elif 'FLOAT' in subtype:
                    bits = 32
            except Exception:
                pass
            return {
                'channels': info.channels,
                'sample_rate': info.samplerate,
                'bits': bits,
                'frames': info.frames,
                'duration': info.duration
            }
        except Exception:
            pass  # Fall through to wave module
    
    # Fallback to wave module
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


def get_lufs(filepath: str) -> Optional[float]:
    """
    Get integrated loudness (LUFS) of audio file using ffmpeg.
    
    Args:
        filepath: Path to audio file
    
    Returns:
        Integrated loudness in LUFS, or None on error
    """
    try:
        cmd = [
            'ffmpeg', '-i', filepath,
            '-af', 'loudnorm=print_format=json',
            '-f', 'null', '-'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse loudnorm output from stderr
        output = result.stderr
        # Find the JSON part
        json_start = output.rfind('{')
        json_end = output.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = output[json_start:json_end]
            data = json.loads(json_str)
            return float(data.get('input_i', -24.0))
    except Exception as e:
        print(f"LUFS measurement error: {e}")
    return None


def normalize_lufs(input_path: str, output_path: str, 
                   target_lufs: float = TARGET_LUFS) -> bool:
    """
    Normalize audio to target LUFS using ffmpeg loudnorm filter.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file
        target_lufs: Target loudness in LUFS (default: -14)
    
    Returns:
        True if successful
    """
    try:
        # Two-pass loudnorm for accurate normalization
        # First pass: measure
        cmd_measure = [
            'ffmpeg', '-i', input_path,
            '-af', 'loudnorm=print_format=json',
            '-f', 'null', '-'
        ]
        result = subprocess.run(cmd_measure, capture_output=True, text=True)
        
        # Parse measurements
        output = result.stderr
        json_start = output.rfind('{')
        json_end = output.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            measurements = json.loads(output[json_start:json_end])
            
            # Second pass: normalize with measured values
            loudnorm_filter = (
                f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:"
                f"measured_I={measurements.get('input_i', -24)}:"
                f"measured_LRA={measurements.get('input_lra', 7)}:"
                f"measured_TP={measurements.get('input_tp', -2)}:"
                f"measured_thresh={measurements.get('input_thresh', -34)}:"
                f"offset={measurements.get('target_offset', 0)}:linear=true"
            )
        else:
            # Fallback to single-pass
            loudnorm_filter = f'loudnorm=I={target_lufs}:TP=-1.5:LRA=11'
        
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-af', loudnorm_filter,
            output_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception as e:
        print(f"LUFS normalization error: {e}")
        return False


def detect_silence_regions(filepath: str, threshold_db: float = SILENCE_THRESHOLD_DB,
                          min_duration: float = MIN_SILENCE_DURATION) -> List[Tuple[float, float]]:
    """
    Detect silence regions in audio file.
    
    Args:
        filepath: Path to audio file
        threshold_db: Silence threshold in dB
        min_duration: Minimum silence duration to detect
    
    Returns:
        List of (start_time, end_time) tuples for silence regions
    """
    silence_regions = []
    
    try:
        cmd = [
            'ffmpeg', '-i', filepath,
            '-af', f'silencedetect=n={threshold_db}dB:d={min_duration}',
            '-f', 'null', '-'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse silencedetect output
        import re
        starts = re.findall(r'silence_start: ([\d.]+)', result.stderr)
        ends = re.findall(r'silence_end: ([\d.]+)', result.stderr)
        
        for start, end in zip(starts, ends):
            silence_regions.append((float(start), float(end)))
            
    except Exception as e:
        print(f"Silence detection error: {e}")
    
    return silence_regions


def convert_to_wav(input_path: str, output_path: str, 
                   sample_rate: int = TRAINING_SAMPLE_RATE, channels: int = DEFAULT_CHANNELS, 
                   bits: int = DEFAULT_BIT_DEPTH) -> bool:
    """
    Convert audio file to WAV format using ffmpeg.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output WAV file
        sample_rate: Target sample rate (default: 22050 for training)
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


def convert_for_recording(input_path: str, output_path: str) -> bool:
    """Convert audio to high-quality format for recording (48kHz, mono, 16-bit)."""
    return convert_to_wav(input_path, output_path, 
                         sample_rate=RECORDING_SAMPLE_RATE,
                         channels=DEFAULT_CHANNELS,
                         bits=DEFAULT_BIT_DEPTH)


def convert_for_training(input_path: str, output_path: str) -> bool:
    """Convert audio to training format (22050Hz, mono, 16-bit)."""
    return convert_to_wav(input_path, output_path,
                         sample_rate=TRAINING_SAMPLE_RATE,
                         channels=DEFAULT_CHANNELS,
                         bits=DEFAULT_BIT_DEPTH)


def trim_silence(input_path: str, output_path: str,
                 threshold_db: float = SILENCE_THRESHOLD_DB, 
                 min_silence_duration: float = MIN_SILENCE_DURATION,
                 target_leading: float = TARGET_LEADING_SILENCE,
                 target_trailing: float = TARGET_TRAILING_SILENCE) -> bool:
    """
    Trim leading and trailing silence from audio file using ffmpeg.
    Maintains consistent silence padding at start and end.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output file
        threshold_db: Silence threshold in dB (default: -40)
        min_silence_duration: Minimum silence duration in seconds
        target_leading: Target silence at beginning (seconds)
        target_trailing: Target silence at end (seconds)
    
    Returns:
        True if successful
    """
    try:
        # First, trim all silence
        temp_path = output_path + '.temp.wav'
        
        # Use ffmpeg silenceremove filter
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-af', f'silenceremove=start_periods=1:start_duration={min_silence_duration}:'
                   f'start_threshold={threshold_db}dB:'
                   f'stop_periods=1:stop_duration={min_silence_duration}:'
                   f'stop_threshold={threshold_db}dB',
            temp_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        
        # Add back consistent padding
        # Get sample rate from temp file
        info = get_audio_info(temp_path)
        if info:
            # Add padding using adelay and apad filters
            delay_ms = int(target_leading * 1000)
            cmd_pad = [
                'ffmpeg', '-y', '-i', temp_path,
                '-af', f'adelay={delay_ms}|{delay_ms},'
                       f'apad=pad_dur={target_trailing}',
                output_path
            ]
            subprocess.run(cmd_pad, capture_output=True, check=True)
            os.remove(temp_path)
        else:
            os.rename(temp_path, output_path)
        
        return True
    except Exception as e:
        print(f"Trim error: {e}")
        # Clean up temp file if exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False


def normalize_loudness(input_path: str, output_path: str,
                       target_lufs: float = TARGET_LUFS) -> bool:
    """
    Normalize audio loudness using ffmpeg loudnorm filter.
    Uses two-pass normalization for accurate results.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file
        target_lufs: Target loudness in LUFS (default: -14)
    
    Returns:
        True if successful
    """
    return normalize_lufs(input_path, output_path, target_lufs)


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
        # Use soundfile if available (handles more formats)
        if HAS_SOUNDFILE:
            samples, _ = sf.read(filepath, dtype='float32')
            if len(samples.shape) > 1:
                samples = samples.mean(axis=1)  # Convert to mono
            peak = float(np.max(np.abs(samples)))
            return (peak >= threshold, peak)
        
        # Fallback to wave module
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
        # Use soundfile if available (handles more formats)
        if HAS_SOUNDFILE:
            samples, _ = sf.read(filepath, dtype='float32')
            if len(samples.shape) > 1:
                samples = samples.mean(axis=1)  # Convert to mono
            
            rms = np.sqrt(np.mean(samples ** 2))
            if rms > 0:
                rms_db = 20 * math.log10(rms)
                return rms_db
            return -100.0
        
        # Fallback to wave module
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


def check_duration(filepath: str, min_dur: float = MIN_DURATION, 
                   max_dur: float = MAX_DURATION) -> Tuple[bool, float]:
    """
    Check if audio duration is within acceptable range.
    
    Args:
        filepath: Path to WAV file
        min_dur: Minimum duration in seconds (default: 0.5)
        max_dur: Maximum duration in seconds (default: 15.0)
    
    Returns:
        Tuple of (is_valid, duration)
    """
    info = get_audio_info(filepath)
    if info is None:
        return (False, 0.0)
    
    duration = info['duration']
    is_valid = min_dur <= duration <= max_dur
    return (is_valid, duration)


def estimate_snr(filepath: str) -> Optional[float]:
    """
    Estimate Signal-to-Noise Ratio of audio file.
    Uses silence detection to estimate noise floor.
    
    Args:
        filepath: Path to WAV file
    
    Returns:
        Estimated SNR in dB, or None on error
    """
    if not HAS_NUMPY:
        return None
    
    try:
        # Use soundfile if available (handles more formats)
        if HAS_SOUNDFILE:
            samples, sample_rate = sf.read(filepath, dtype='float32')
            if len(samples.shape) > 1:
                samples = samples.mean(axis=1)  # Convert to mono
        else:
            # Fallback to wave module
            with wave.open(filepath, 'rb') as w:
                frames = w.readframes(w.getnframes())
                sample_rate = w.getframerate()
                
                if w.getsampwidth() == 2:
                    samples = np.frombuffer(frames, dtype=np.int16).astype(float)
                    samples /= 32767.0
                else:
                    samples = np.frombuffer(frames, dtype=np.uint8).astype(float)
                    samples = (samples - 128) / 128.0
        
        # Calculate RMS in windows
        window_size = int(0.025 * sample_rate)  # 25ms windows
        hop_size = int(0.010 * sample_rate)  # 10ms hop
        
        rms_values = []
        for i in range(0, len(samples) - window_size, hop_size):
            window = samples[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))
            if rms > 0:
                rms_values.append(20 * math.log10(rms))
        
        if not rms_values:
            return None
        
        # Estimate noise floor as 10th percentile
        noise_floor = np.percentile(rms_values, 10)
        # Estimate signal as 90th percentile
        signal_level = np.percentile(rms_values, 90)
        
        snr = signal_level - noise_floor
        return float(snr)
        
    except Exception as e:
        print(f"SNR estimation error: {e}")
        return None


def check_silence_only(filepath: str, threshold_ratio: float = 0.95) -> bool:
    """
    Check if audio is mostly silence (recording failed).
    
    Args:
        filepath: Path to WAV file
        threshold_ratio: Ratio of silence to total duration to consider as silence-only
    
    Returns:
        True if file is mostly silence
    """
    silence_regions = detect_silence_regions(filepath)
    info = get_audio_info(filepath)
    
    if not info or info['duration'] == 0:
        return True
    
    total_silence = sum(end - start for start, end in silence_regions)
    silence_ratio = total_silence / info['duration']
    
    return silence_ratio >= threshold_ratio


def check_background_noise(filepath: str, max_noise_db: float = -50.0) -> Tuple[bool, Optional[float]]:
    """
    Check for excessive background noise/hiss.
    
    Args:
        filepath: Path to WAV file
        max_noise_db: Maximum acceptable noise floor in dB
    
    Returns:
        Tuple of (has_excessive_noise, estimated_noise_floor)
    """
    snr = estimate_snr(filepath)
    if snr is None:
        return (False, None)
    
    # If SNR is too low, there's excessive noise
    has_excessive_noise = snr < MIN_SNR_DB
    
    return (has_excessive_noise, snr)


def quality_check(filepath: str, strict: bool = False) -> Dict:
    """
    Run comprehensive quality checks on an audio file.
    
    Args:
        filepath: Path to WAV file
        strict: If True, apply stricter thresholds
    
    Returns:
        Dictionary with check results
    """
    results = {
        'filepath': filepath,
        'exists': os.path.exists(filepath),
        'checks': {},
        'issues': []
    }
    
    if not results['exists']:
        results['issues'].append('File does not exist')
        return results
    
    # Get info
    info = get_audio_info(filepath)
    results['info'] = info
    
    # Duration check
    min_dur = MIN_DURATION if not strict else 0.8
    max_dur = MAX_DURATION if not strict else 10.0
    dur_valid, duration = check_duration(filepath, min_dur, max_dur)
    results['checks']['duration'] = {
        'valid': dur_valid,
        'value': duration,
        'unit': 'seconds',
        'threshold': f'{min_dur}-{max_dur}s'
    }
    if not dur_valid:
        if duration < min_dur:
            results['issues'].append(f'Too short ({duration:.2f}s < {min_dur}s)')
        else:
            results['issues'].append(f'Too long ({duration:.2f}s > {max_dur}s)')
    
    # Clipping check
    has_clipping, peak = check_clipping(filepath, CLIPPING_THRESHOLD)
    results['checks']['clipping'] = {
        'valid': not has_clipping,
        'value': peak,
        'unit': 'peak_ratio',
        'threshold': CLIPPING_THRESHOLD
    }
    if has_clipping:
        results['issues'].append(f'Audio is clipping (peak={peak:.3f})')
    
    # RMS check
    rms = calculate_rms(filepath)
    rms_valid = rms is not None and MIN_RMS_DB <= rms <= MAX_RMS_DB
    results['checks']['rms'] = {
        'valid': rms_valid,
        'value': rms,
        'unit': 'dB',
        'threshold': f'{MIN_RMS_DB} to {MAX_RMS_DB} dB'
    }
    if not rms_valid and rms is not None:
        if rms < MIN_RMS_DB:
            results['issues'].append(f'Volume too low (RMS={rms:.1f}dB)')
        else:
            results['issues'].append(f'Volume too high (RMS={rms:.1f}dB)')
    
    # SNR check
    snr = estimate_snr(filepath)
    snr_valid = snr is not None and snr >= MIN_SNR_DB
    results['checks']['snr'] = {
        'valid': snr_valid,
        'value': snr,
        'unit': 'dB',
        'threshold': f'>= {MIN_SNR_DB} dB'
    }
    if not snr_valid and snr is not None:
        results['issues'].append(f'Too much background noise (SNR={snr:.1f}dB)')
    
    # Silence-only check
    is_silence_only = check_silence_only(filepath)
    results['checks']['silence_only'] = {
        'valid': not is_silence_only,
        'value': is_silence_only
    }
    if is_silence_only:
        results['issues'].append('Recording is mostly silence')
    
    # Overall result
    results['passed'] = all(
        c.get('valid', True) for c in results['checks'].values()
    )
    
    return results


def filter_low_quality(input_dir: str, output_dir: str = None,
                       strict: bool = False) -> Tuple[List[str], List[str]]:
    """
    Filter out low quality audio files.
    
    Args:
        input_dir: Directory with audio files
        output_dir: Optional directory to copy good files to
        strict: Apply stricter quality thresholds
    
    Returns:
        Tuple of (good_files, bad_files)
    """
    good_files = []
    bad_files = []
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    extensions = {'.wav', '.mp3', '.webm', '.ogg', '.m4a', '.flac'}
    
    for filename in os.listdir(input_dir):
        ext = Path(filename).suffix.lower()
        if ext not in extensions:
            continue
        
        filepath = os.path.join(input_dir, filename)
        result = quality_check(filepath, strict=strict)
        
        if result['passed']:
            good_files.append(filepath)
            if output_dir:
                import shutil
                shutil.copy2(filepath, os.path.join(output_dir, filename))
        else:
            bad_files.append({
                'file': filepath,
                'issues': result['issues']
            })
    
    return good_files, bad_files


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def speed_perturbation(input_path: str, output_path: str, factor: float) -> bool:
    """
    Apply speed perturbation to audio.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file
        factor: Speed factor (0.95 = slower, 1.05 = faster)
    
    Returns:
        True if successful
    """
    try:
        # Use ffmpeg atempo filter
        # atempo only accepts 0.5 to 2.0, chain for more extreme values
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-af', f'atempo={factor}',
            output_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception as e:
        print(f"Speed perturbation error: {e}")
        return False


def volume_perturbation(input_path: str, output_path: str, 
                        factor: float) -> bool:
    """
    Apply volume scaling to audio.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file
        factor: Volume factor (0.9 = quieter, 1.1 = louder)
    
    Returns:
        True if successful
    """
    try:
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-af', f'volume={factor}',
            output_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception as e:
        print(f"Volume perturbation error: {e}")
        return False


def generate_augmented_dataset(input_dir: str, output_dir: str,
                               speed_factors: List[float] = None,
                               volume_factors: List[float] = None) -> Dict:
    """
    Generate augmented versions of audio files.
    
    Args:
        input_dir: Directory with original audio files
        output_dir: Directory to save augmented files
        speed_factors: List of speed factors to apply (e.g., [0.95, 1.05])
        volume_factors: List of volume factors to apply (e.g., [0.9, 1.1])
    
    Returns:
        Dictionary with augmentation statistics
    """
    if speed_factors is None:
        speed_factors = [0.95, 1.05]
    if volume_factors is None:
        volume_factors = [0.9, 1.1]
    
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {
        'original_files': 0,
        'speed_augmented': 0,
        'volume_augmented': 0,
        'total_output': 0
    }
    
    extensions = {'.wav'}
    
    for filename in os.listdir(input_dir):
        ext = Path(filename).suffix.lower()
        if ext not in extensions:
            continue
        
        input_path = os.path.join(input_dir, filename)
        base_name = Path(filename).stem
        stats['original_files'] += 1
        
        # Copy original
        import shutil
        shutil.copy2(input_path, os.path.join(output_dir, filename))
        stats['total_output'] += 1
        
        # Speed augmentation
        for factor in speed_factors:
            if factor == 1.0:
                continue
            out_name = f"{base_name}_speed{factor:.2f}.wav"
            out_path = os.path.join(output_dir, out_name)
            if speed_perturbation(input_path, out_path, factor):
                stats['speed_augmented'] += 1
                stats['total_output'] += 1
        
        # Volume augmentation
        for factor in volume_factors:
            if factor == 1.0:
                continue
            out_name = f"{base_name}_vol{factor:.2f}.wav"
            out_path = os.path.join(output_dir, out_name)
            if volume_perturbation(input_path, out_path, factor):
                stats['volume_augmented'] += 1
                stats['total_output'] += 1
    
    return stats


# ============================================================================
# METADATA UTILITIES
# ============================================================================

def compute_file_hash(filepath: str) -> str:
    """Compute MD5 hash of file for duplicate detection."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_duplicates(directory: str) -> Dict[str, List[str]]:
    """
    Find duplicate audio files by content hash.
    
    Args:
        directory: Directory to scan
    
    Returns:
        Dictionary mapping hash to list of duplicate file paths
    """
    hash_to_files: Dict[str, List[str]] = {}
    
    extensions = {'.wav', '.mp3', '.webm', '.ogg', '.m4a', '.flac'}
    
    for filename in os.listdir(directory):
        ext = Path(filename).suffix.lower()
        if ext not in extensions:
            continue
        
        filepath = os.path.join(directory, filename)
        file_hash = compute_file_hash(filepath)
        
        if file_hash not in hash_to_files:
            hash_to_files[file_hash] = []
        hash_to_files[file_hash].append(filepath)
    
    # Return only duplicates
    return {h: files for h, files in hash_to_files.items() if len(files) > 1}


def normalize_text_utf8(text: str) -> str:
    """
    Normalize text to NFC Unicode form.
    
    Args:
        text: Input text
    
    Returns:
        NFC-normalized text
    """
    return unicodedata.normalize('NFC', text)


def validate_metadata_entry(entry: Dict, wav_dir: str) -> Dict:
    """
    Validate a metadata entry.
    
    Args:
        entry: Metadata dictionary with filename, text, etc.
        wav_dir: Directory containing WAV files
    
    Returns:
        Dictionary with validation results
    """
    result = {
        'valid': True,
        'issues': []
    }
    
    # Check filename exists
    filename = entry.get('filename', '')
    if filename:
        filepath = os.path.join(wav_dir, filename)
        if not os.path.exists(filepath):
            result['valid'] = False
            result['issues'].append(f'Audio file not found: {filename}')
        else:
            # Get actual duration
            info = get_audio_info(filepath)
            if info:
                entry['actual_duration'] = info['duration']
    
    # Check text
    text = entry.get('text', '')
    if not text:
        result['valid'] = False
        result['issues'].append('Empty text')
    else:
        # Normalize and check
        normalized = normalize_text_utf8(text)
        if normalized != text:
            result['issues'].append('Text needs UTF-8 normalization')
            entry['normalized_text'] = normalized
    
    return result


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
    
    # Main operations
    parser.add_argument('--process', action='store_true', help='Process recordings directory')
    parser.add_argument('--filter', action='store_true', help='Filter out low quality files')
    parser.add_argument('--augment', action='store_true', help='Generate augmented dataset')
    parser.add_argument('--check', type=str, help='Run quality check on single file')
    parser.add_argument('--find-duplicates', action='store_true', help='Find duplicate audio files')
    
    # Directories
    parser.add_argument('--input', type=str, default='dataset/recordings', help='Input directory')
    parser.add_argument('--output', type=str, default='dataset/processed/wav_22k_mono', help='Output directory')
    
    # Processing options
    parser.add_argument('--sample-rate', type=int, default=TRAINING_SAMPLE_RATE, help='Target sample rate')
    parser.add_argument('--target-lufs', type=float, default=TARGET_LUFS, help='Target loudness in LUFS')
    parser.add_argument('--no-trim', action='store_true', help='Skip silence trimming')
    parser.add_argument('--no-normalize', action='store_true', help='Skip loudness normalization')
    parser.add_argument('--strict', action='store_true', help='Apply strict quality thresholds')
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.check:
        result = quality_check(args.check, strict=args.strict)
        print(json.dumps(result, indent=2))
        
    elif args.find_duplicates:
        input_dir = os.path.join(base_dir, args.input)
        print(f"\nScanning for duplicates in: {input_dir}")
        duplicates = find_duplicates(input_dir)
        
        if duplicates:
            print(f"\nFound {len(duplicates)} sets of duplicates:")
            for file_hash, files in duplicates.items():
                print(f"\n  Hash: {file_hash[:8]}...")
                for f in files:
                    print(f"    - {Path(f).name}")
        else:
            print("No duplicates found.")
            
    elif args.filter:
        input_dir = os.path.join(base_dir, args.input)
        output_dir = os.path.join(base_dir, args.output, 'filtered')
        
        print(f"\nFiltering low quality files...")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Strict mode: {args.strict}")
        print()
        
        good_files, bad_files = filter_low_quality(input_dir, output_dir, strict=args.strict)
        
        print(f"\nResults:")
        print(f"  Passed: {len(good_files)}")
        print(f"  Failed: {len(bad_files)}")
        
        if bad_files:
            print(f"\nRejected files:")
            for item in bad_files[:10]:  # Show first 10
                print(f"  {Path(item['file']).name}: {', '.join(item['issues'])}")
            if len(bad_files) > 10:
                print(f"  ... and {len(bad_files) - 10} more")
                
    elif args.augment:
        input_dir = os.path.join(base_dir, args.input)
        output_dir = os.path.join(base_dir, args.output, 'augmented')
        
        print(f"\nGenerating augmented dataset...")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        print()
        
        stats = generate_augmented_dataset(
            input_dir, output_dir,
            speed_factors=[0.95, 1.0, 1.05],
            volume_factors=[0.9, 1.0, 1.1]
        )
        
        print(f"\nAugmentation results:")
        print(f"  Original files: {stats['original_files']}")
        print(f"  Speed augmented: {stats['speed_augmented']}")
        print(f"  Volume augmented: {stats['volume_augmented']}")
        print(f"  Total output: {stats['total_output']}")
        
    elif args.process:
        input_dir = os.path.join(base_dir, args.input)
        output_dir = os.path.join(base_dir, args.output)
        
        print(f"\nProcessing audio files...")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Sample rate: {args.sample_rate}")
        print(f"Target LUFS: {args.target_lufs}")
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
