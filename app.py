#!/usr/bin/env python3
"""
Bengali TTS Recording Flask Application

Web-based audio recorder for collecting voice samples.
Features:
- Browser-based recording using MediaRecorder API
- Real-time audio quality checks
- Prompt navigation and status tracking
- Metadata management
"""

import os
import csv
import json
import wave
import subprocess
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
RECORDINGS_DIR = os.path.join(DATASET_DIR, 'recordings')
PROMPTS_FILE = os.path.join(DATASET_DIR, 'prompts', 'prompts.csv')
METADATA_FILE = os.path.join(RECORDINGS_DIR, 'metadata.jsonl')

# Create directories
os.makedirs(RECORDINGS_DIR, exist_ok=True)


def load_prompts():
    """Load prompts from CSV file."""
    prompts = []
    if os.path.exists(PROMPTS_FILE):
        with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            prompts = list(reader)
    return prompts


def save_prompts(prompts):
    """Save prompts to CSV file."""
    if prompts:
        with open(PROMPTS_FILE, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['prompt_id', 'type', 'text', 'status'])
            writer.writeheader()
            writer.writerows(prompts)


def get_audio_duration(filepath):
    """Get duration of audio file in seconds."""
    try:
        with wave.open(filepath, 'rb') as w:
            frames = w.getnframes()
            rate = w.getframerate()
            return frames / float(rate)
    except Exception:
        return None


def is_safe_path(base_dir, path):
    """Check if path is safe (no path traversal)."""
    # Resolve to absolute paths
    base = os.path.abspath(base_dir)
    target = os.path.abspath(path)
    # Ensure target is within base directory
    return target.startswith(base + os.sep) or target == base


def convert_to_wav(input_path, output_path):
    """Convert audio file to WAV format (16kHz, mono, 16-bit)."""
    # Validate paths to prevent command injection
    if not is_safe_path(DATASET_DIR, input_path) or not is_safe_path(DATASET_DIR, output_path):
        print(f"Security error: Invalid path")
        return False
    
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', input_path,
            '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le',
            output_path
        ], capture_output=True, check=True)
        return True
    except Exception as e:
        print(f"Conversion error: {e}")
        return False


def append_metadata(metadata):
    """Append metadata to JSONL file."""
    with open(METADATA_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(metadata, ensure_ascii=False) + '\n')


@app.route('/')
def index():
    """Render the recording interface."""
    return render_template('recorder.html')


@app.route('/api/prompts')
def get_prompts():
    """Get all prompts."""
    prompts = load_prompts()
    return jsonify({
        'ok': True,
        'prompts': prompts,
        'total': len(prompts)
    })


@app.route('/api/prompts/<prompt_id>')
def get_prompt(prompt_id):
    """Get a specific prompt."""
    prompts = load_prompts()
    for prompt in prompts:
        if prompt['prompt_id'] == prompt_id:
            return jsonify({'ok': True, 'prompt': prompt})
    return jsonify({'ok': False, 'error': 'Prompt not found'}), 404


@app.route('/api/prompts/<prompt_id>/status', methods=['POST'])
def update_prompt_status(prompt_id):
    """Update prompt status."""
    data = request.get_json()
    new_status = data.get('status', 'pending')
    
    prompts = load_prompts()
    for prompt in prompts:
        if prompt['prompt_id'] == prompt_id:
            prompt['status'] = new_status
            save_prompts(prompts)
            return jsonify({'ok': True, 'prompt': prompt})
    
    return jsonify({'ok': False, 'error': 'Prompt not found'}), 404


@app.route('/api/save', methods=['POST'])
def save_audio():
    """Save uploaded audio recording."""
    speaker = request.form.get('speaker', 'S01')
    prompt_id = request.form.get('prompt_id', '00000')
    text = request.form.get('text', '')
    prompt_type = request.form.get('type', 'unknown')
    
    audio_file = request.files.get('audio_data')
    if not audio_file:
        return jsonify({'ok': False, 'error': 'No audio data received'})
    
    # Generate filename
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    base_filename = f"{speaker}_{timestamp}_{prompt_id}"
    
    # Save original file (usually webm/ogg from browser)
    temp_path = os.path.join(RECORDINGS_DIR, f"{base_filename}_temp.webm")
    wav_path = os.path.join(RECORDINGS_DIR, f"{base_filename}.wav")
    
    audio_file.save(temp_path)
    
    # Convert to WAV
    if convert_to_wav(temp_path, wav_path):
        # Remove temp file
        os.remove(temp_path)
        audio_path = wav_path
        filename = f"{base_filename}.wav"
    else:
        # Keep original if conversion fails
        audio_path = temp_path
        filename = f"{base_filename}_temp.webm"
    
    # Get duration
    duration = get_audio_duration(audio_path) if audio_path.endswith('.wav') else None
    
    # Create metadata
    metadata = {
        'speaker_id': speaker,
        'prompt_id': prompt_id,
        'text': text,
        'type': prompt_type,
        'filename': filename,
        'duration': duration,
        'sample_rate': 16000 if audio_path.endswith('.wav') else None,
        'bits': 16 if audio_path.endswith('.wav') else None,
        'accepted': True,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    # Save metadata
    append_metadata(metadata)
    
    # Update prompt status
    prompts = load_prompts()
    for prompt in prompts:
        if prompt['prompt_id'] == prompt_id:
            prompt['status'] = 'done'
            break
    save_prompts(prompts)
    
    return jsonify({
        'ok': True,
        'metadata': metadata,
        'message': f'Saved as {filename}'
    })


@app.route('/api/stats')
def get_stats():
    """Get recording statistics."""
    prompts = load_prompts()
    
    stats = {
        'total_prompts': len(prompts),
        'pending': sum(1 for p in prompts if p.get('status') == 'pending'),
        'done': sum(1 for p in prompts if p.get('status') == 'done'),
        'retry': sum(1 for p in prompts if p.get('status') == 'retry'),
    }
    
    # Count recordings
    recordings = [f for f in os.listdir(RECORDINGS_DIR) if f.endswith('.wav')]
    stats['recordings'] = len(recordings)
    
    # Calculate total duration
    total_duration = 0
    for rec in recordings:
        path = os.path.join(RECORDINGS_DIR, rec)
        dur = get_audio_duration(path)
        if dur:
            total_duration += dur
    
    stats['total_duration_seconds'] = round(total_duration, 2)
    stats['total_duration_minutes'] = round(total_duration / 60, 2)
    
    return jsonify({'ok': True, 'stats': stats})


@app.route('/recordings/<filename>')
def serve_recording(filename):
    """Serve a recording file."""
    return send_from_directory(RECORDINGS_DIR, filename)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Bengali TTS Recording Server")
    print("=" * 60)
    print(f"\nDataset directory: {DATASET_DIR}")
    print(f"Recordings directory: {RECORDINGS_DIR}")
    print(f"Prompts file: {PROMPTS_FILE}")
    
    # Check if prompts exist
    prompts = load_prompts()
    if not prompts:
        print("\n⚠️  No prompts found! Run 'python generate_prompts.py' first.")
    else:
        print(f"\n✓ Loaded {len(prompts)} prompts")
    
    print("\nStarting server at http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    print("=" * 60 + "\n")
    
    # Note: Set debug=False in production environments
    import sys
    debug_mode = '--debug' in sys.argv
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)
