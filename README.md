# Bengali TTS Custom - Voice Recording & Training System

A complete system for creating your own Bengali Text-to-Speech model with your own Bangladeshi accent.

## Features

- 🎤 **Web-based Recording**: Browser-based audio recorder with real-time quality checks
- 📝 **Complete Bengali Phoneme Coverage**: All vowels, consonants, matras, and 150+ conjuncts (juktakkhor)
- 🔤 **G2P Engine**: Convert Bengali text to phoneme sequences with number expansion
- 🎯 **Smart Prompts**: Automatically generated prompts for balanced phoneme coverage
- 🔊 **Audio QA**: Automatic quality checks (duration, loudness, clipping, noise detection)
- 🚀 **GPU Accelerated**: CUDA support for training (GTX 1660 Super compatible)
- 📊 **Training Stability**: Warmup, cosine decay, gradient accumulation, AMP, early stopping
- 🎚️ **Waveform Preview**: Real-time audio visualization and level metering

---

## Table of Contents

1. [Setup](#setup)
2. [Recording](#recording)
3. [Preprocessing](#preprocessing)
4. [Training](#training)
5. [Inference](#inference)
6. [Project Structure](#project-structure)
7. [Configuration](#configuration)

---

## Setup

### 1. System Requirements

- **Python**: 3.8+
- **GPU**: GTX 1660 Super (6GB) or better (for training)
- **ffmpeg**: Required for audio processing
- **OS**: Linux/Windows/macOS

### 2. Install Dependencies

```bash
# Clone the repository
cd bengali_tts_custom

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install Python dependencies
pip install -r requirements.txt

# Install ffmpeg (if not already installed)
# Ubuntu/Debian:
sudo apt install ffmpeg
# macOS:
brew install ffmpeg
# Windows: Download from https://ffmpeg.org/download.html
```

### 3. Install PyTorch with CUDA (for training)

```bash
# For GTX 1660 Super, use CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Generate Recording Prompts

```bash
python generate_prompts.py
```

This creates `dataset/prompts/prompts.csv` with comprehensive Bengali phoneme coverage.

---

## Recording

### Starting the Recording Server

```bash
python app.py
```

Open http://localhost:5000 in your browser.

### Recording Interface Features

- **Real-time Level Meter**: Shows microphone input level
- **Waveform Preview**: Visualizes recorded audio
- **Quality Checks**: Automatic detection of:
  - Silence-only recordings
  - Audio clipping
  - Low volume
  - Background noise
- **Navigation**: Previous/Next buttons, go-to specific prompt
- **History Preview**: See previous and next prompts
- **Keyboard Shortcuts**:
  - `Space`: Start/Stop recording
  - `Enter`: Accept recording
  - `R`: Re-record
  - `S`: Skip prompt
  - `←/→`: Navigate prompts

### Recording Tips

1. **Environment**: Record in a quiet room with minimal echo
2. **Microphone**: Use a decent microphone, maintain consistent distance (~15-20cm)
3. **Consistency**: Keep the same speaking style, pace, and volume throughout
4. **Sessions**: Record in 30-minute sessions to avoid fatigue
5. **Audio Format**: Recordings are saved as 48kHz, mono, 16-bit PCM WAV

---

## Preprocessing

### Process Recorded Audio

```bash
# Full processing pipeline (normalize, trim silence)
python audio_utils.py --process --input dataset/recordings --output dataset/processed/wav_22k_mono

# With specific settings
python audio_utils.py --process --sample-rate 22050 --target-lufs -14

# Filter out low-quality recordings
python audio_utils.py --filter --input dataset/recordings --output dataset/filtered --strict

# Generate augmented dataset (speed/volume perturbation)
python audio_utils.py --augment --input dataset/processed/wav_22k_mono --output dataset/augmented

# Check single file quality
python audio_utils.py --check path/to/audio.wav

# Find duplicate recordings
python audio_utils.py --find-duplicates --input dataset/recordings
```

### Prepare Training Data

```bash
python prepare_training.py
```

This creates:
- `dataset/metadata.csv` - LJSpeech format metadata
- `dataset/train.csv`, `val.csv`, `test.csv` - Data splits
- `dataset/train_filelist.txt` - Full paths for training

---

## Training

### Training Configuration

The training config (`training_config.py`) includes:

- **Learning Rate**: Reduced 3-5x for stability (generator: 1e-4, discriminator: 5e-5)
- **Warmup**: 4000 steps with cosine decay
- **Batch Size**: 8 (with 4x gradient accumulation = effective 32)
- **AMP**: Automatic Mixed Precision for memory efficiency
- **Checkpointing**: Auto-save every 1000 steps with rollback support
- **Early Stopping**: Stop when validation loss plateaus or alignment degrades
- **Augmentation**: Auto-disable after 35% of training

### Start Training

```bash
# Generate default config
python training_config.py

# View/edit training_config.json, then start training
python trainer.py --config training_config.json
```

### Monitor Training

The trainer logs:
- Attention alignment plots (every 500 steps)
- Mel spectrogram previews (every 500 steps)
- Audio samples (every 1000 steps)
- Training/validation losses

Check `output/attention_plots/` and `output/mel_plots/` for visualizations.

---

## Inference

### Text-to-Speech Synthesis

```bash
# Basic synthesis
python inference.py --text "আমার সোনার বাংলা" --output output.wav

# With specific model
python inference.py --text "Hello" --model checkpoints/best_checkpoint.pt --output hello.wav

# Phoneme-only mode (no synthesis)
python inference.py --text "বাংলাদেশ" --phonemes-only

# Export model
python inference.py --model checkpoints/best.pt --export-onnx model.onnx
python inference.py --model checkpoints/best.pt --export-torchscript model.pt
```

### G2P Testing

```bash
# Run G2P tests
python g2p.py

# Convert specific text
python g2p.py "আমার নাম বাংলা"
```

---

## Project Structure

```
bengali_tts_custom/
├── app.py                  # Flask recording server
├── g2p.py                  # Grapheme-to-Phoneme engine
├── generate_prompts.py     # Prompt generation script
├── audio_utils.py          # Audio processing utilities
├── prepare_training.py     # Training data preparation
├── training_config.py      # Training configuration
├── trainer.py              # Training pipeline
├── inference.py            # Inference CLI
├── requirements.txt        # Python dependencies
├── templates/
│   └── recorder.html       # Recording UI
├── static/
│   └── style.css           # UI styling
└── dataset/
    ├── recordings/         # Raw recordings (48kHz)
    ├── prompts/            # Generated prompts
    ├── processed/          # Processed audio (22050Hz)
    └── metadata.csv        # Training metadata
```

---

## Configuration

### Audio Settings

| Setting | Recording | Training |
|---------|-----------|----------|
| Sample Rate | 48kHz | 22050Hz |
| Channels | Mono | Mono |
| Bit Depth | 16-bit | 16-bit |
| Target LUFS | -14 dB | -14 dB |

### Training Settings (GTX 1660 Super Optimized)

| Setting | Value |
|---------|-------|
| Batch Size | 8 |
| Gradient Accumulation | 4 (effective: 32) |
| Generator LR | 1e-4 |
| Discriminator LR | 5e-5 |
| Warmup Steps | 4000 |
| AMP | Enabled |
| Checkpoint Interval | 1000 steps |

### Quality Thresholds

| Check | Threshold |
|-------|-----------|
| Min Duration | 0.5 seconds |
| Max Duration | 15 seconds |
| Min RMS | -35 dB |
| Max RMS | -10 dB |
| Clipping | Peak < 0.99 |
| SNR | > 20 dB |

---

## Bengali Phoneme Inventory

### Vowels (স্বরবর্ণ) - 11
অ, আ, ই, ঈ, উ, ঊ, ঋ, এ, ঐ, ও, ঔ

### Consonants (ব্যঞ্জনবর্ণ) - 39
ক, খ, গ, ঘ, ঙ, চ, ছ, জ, ঝ, ঞ, ট, ঠ, ড, ঢ, ণ, ত, থ, দ, ধ, ন, প, ফ, ব, ভ, ম, য, র, ল, শ, ষ, স, হ, ড়, ঢ়, য়, ৎ, ং, ঃ, ঁ

### Matras (মাত্রা) - 10
া, ি, ী, ু, ূ, ৃ, ে, ৈ, ো, ৌ

### Conjuncts (যুক্তাক্ষর)
The G2P engine supports 150+ conjuncts including:
- Ya-phala (য-ফলা): ক্য, গ্য, ব্য...
- Ra-phala (র-ফলা): ক্র, প্র, শ্র...
- Geminates: ক্ক, ত্ত, ন্ন, ম্ম...
- Special: ক্ষ, জ্ঞ, ঞ্চ, ঞ্জ...

---

## Dataset Size Recommendations

| Quality | Duration | Utterances | Result |
|---------|----------|------------|--------|
| Minimum | 1-2 hours | ~1,000 | Robotic but usable |
| Good | 4-6 hours | ~5,000 | Natural sounding |
| Production | 10+ hours | ~20,000 | Studio quality |

---

## Troubleshooting

### Common Issues

1. **Microphone not detected**: Check browser permissions
2. **ffmpeg not found**: Install ffmpeg and add to PATH
3. **CUDA out of memory**: Reduce batch_size in training_config.py
4. **Training collapse**: Check attention plots, try rollback checkpoint
5. **Poor audio quality**: Use --strict flag when filtering

### Getting Help

If you encounter issues, check:
1. Console output for error messages
2. Training logs in `output/` directory
3. Audio quality using `audio_utils.py --check`

---

## License

MIT License - Feel free to use and modify for your projects.
