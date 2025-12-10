# Bengali TTS Custom - Voice Recording & Training System

A complete system for creating your own Bengali Text-to-Speech model with:
- Flask-based web recorder for collecting voice samples
- Grapheme-to-Phoneme (G2P) engine for Bengali
- Prompt generation for comprehensive phoneme coverage
- Audio processing utilities
- Training pipeline integration

## Features

- 🎤 **Web-based Recording**: Browser-based audio recorder with real-time quality checks
- 📝 **Complete Bengali Phoneme Coverage**: Vowels, consonants, matras, and 150+ conjuncts (juktakkhor)
- 🔤 **G2P Engine**: Convert Bengali text to phoneme sequences
- 🎯 **Smart Prompts**: Automatically generated prompts for balanced phoneme coverage
- 🔊 **Audio QA**: Automatic quality checks (duration, loudness, clipping detection)
- 🚀 **GPU Accelerated**: CUDA support for training (GTX 1660 Super compatible)

## Quick Start

### 1. Install Dependencies

```bash
cd bengali_tts_custom
pip install -r requirements.txt
```

### 2. Generate Recording Prompts

```bash
python generate_prompts.py
```

This creates `dataset/prompts/prompts.csv` with:
- All vowels (shoroborno)
- All consonants with matras
- 150+ common conjuncts (juktakkhor)
- Carrier sentences

### 3. Start the Recording App

```bash
python app.py
```

Open http://localhost:5000 in your browser to start recording.

### 4. Record Your Voice

1. Enter your Speaker ID
2. Click "Record" and speak the displayed prompt
3. Click "Stop" when done
4. Listen to playback, then Accept or Re-record
5. Navigate through all prompts

### 5. Process & Train

After collecting enough recordings (minimum 1-2 hours):

```bash
# Process audio files
python audio_utils.py --process

# Generate training metadata
python prepare_training.py
```

## Project Structure

```
bengali_tts_custom/
├── app.py                 # Flask recorder application
├── g2p.py                 # Grapheme-to-Phoneme engine
├── generate_prompts.py    # Prompt generation script
├── audio_utils.py         # Audio processing utilities
├── prepare_training.py    # Training data preparation
├── requirements.txt       # Python dependencies
├── templates/
│   └── recorder.html      # Recording UI
├── static/
│   └── style.css          # UI styling
└── dataset/
    ├── recordings/        # Raw recordings
    ├── prompts/           # Generated prompts
    ├── processed/         # Processed audio
    ├── train/             # Training split
    ├── val/               # Validation split
    └── test/              # Test split
```

## Bengali Phoneme Inventory

### Vowels (স্বরবর্ণ) - 11
অ, আ, ই, ঈ, উ, ঊ, ঋ, এ, ঐ, ও, ঔ

### Consonants (ব্যঞ্জনবর্ণ) - 39
ক, খ, গ, ঘ, ঙ, চ, ছ, জ, ঝ, ঞ, ট, ঠ, ড, ঢ, ণ, ত, থ, দ, ধ, ন, প, ফ, ব, ভ, ম, য, র, ল, শ, ষ, স, হ, ড়, ঢ়, য়, ৎ, ং, ঃ, ঁ

### Matras (মাত্রা) - 10
া, ি, ী, ু, ূ, ৃ, ে, ৈ, ো, ৌ

## Recording Tips

1. **Environment**: Record in a quiet room with minimal echo
2. **Microphone**: Use a decent microphone, maintain consistent distance
3. **Consistency**: Keep the same speaking style throughout
4. **Pace**: Speak naturally, not too fast or slow
5. **Sessions**: Record in 30-minute sessions to avoid fatigue

## Dataset Size Recommendations

| Quality Level | Duration | Utterances | Result |
|---------------|----------|------------|--------|
| Minimum | 1-2 hours | ~1,000 | Robotic but usable |
| Good | 4-6 hours | ~5,000 | Natural sounding |
| Production | 10+ hours | ~20,000 | Studio quality |

## Future: Emotion Support

Planned emotion categories for future versions:
- 😊 Happy/Joyful
- 😢 Sad/Melancholic  
- 😠 Angry
- 😨 Fearful
- 🤗 Warm/Affectionate
- 😐 Neutral
- 😲 Surprised
- 🤔 Thoughtful

## License

MIT License - Feel free to use and modify for your projects.
