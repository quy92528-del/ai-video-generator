# 🎬 AI Video Generator Suite

> Generate **500+ videos** automatically from a single text input using Google Gemini, Stable Diffusion, Runway Gen-3, ElevenLabs, and more.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ Feature Overview

| Feature | Description |
|---------|-------------|
| 🤖 **AI Script Generation** | Google Gemini 1.5 Pro writes scripts, titles, and hashtags |
| 🖼️ **Image Generation** | Stable Diffusion XL via Replicate |
| 🎬 **Video Generation** | Runway Gen-3 (image-to-video & text-to-video) |
| 🧑‍💼 **Avatar Videos** | HeyGen AI presenter with lip-sync |
| 🔊 **TTS & Voice** | Google Cloud TTS + ElevenLabs multilingual |
| 🎵 **Audio Mixing** | Automatic ducking, background music, SFX |
| 🎭 **Face Consistency** | InsightFace + MediaPipe identity verification |
| ✂️ **Post-Production** | MoviePy assembly, captions, colour grading |
| 📐 **Multi-Platform** | 9:16 (TikTok/Reels), 16:9 (YouTube), 1:1 (Instagram) |
| ⚡ **Parallel Processing** | Thread-pool + optional Celery/Redis workers |
| 💰 **Budget Tracking** | Real-time cost monitoring with $300 budget guard |
| 🌍 **7 Languages** | VI, EN, ZH, JA, KO, ES, FR |

---

## 📁 Project Structure

```
ai-video-generator/
├── main.py                 # Streamlit UI – entry point
├── config.py               # Configuration & settings
├── brain_module.py         # Gemini script generation
├── visual_engine.py        # Image & video generation
├── audio_module.py         # TTS & audio mixing
├── post_production.py      # Video assembly & editing
├── face_consistency.py     # Face detection & consistency
├── api_client.py           # Unified API client
├── parallel_processor.py   # Task queue & workers
├── cost_tracker.py         # Budget & cost tracking
├── utils/
│   ├── __init__.py
│   ├── logger.py           # Structured logging
│   ├── validators.py       # Input validation
│   └── helpers.py          # Utility functions
├── requirements.txt
├── .env.example            # API key template
├── setup.py                # Installation script
└── README.md

assets/
├── music/                  # Background music library
├── sounds/                 # SFX library
├── fonts/                  # Custom fonts
└── templates/              # Video templates

output/
├── videos/                 # Generated videos (.mp4)
├── thumbnails/             # Auto-generated thumbnails
└── metadata/               # JSON metadata per video
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python **3.10+**
- `pip` (Python package manager)
- **Optional**: Redis (for distributed workers), ffmpeg (for MoviePy)

```bash
# Check Python version
python --version    # Should be 3.10+

# Install ffmpeg (needed by MoviePy)
# Ubuntu/Debian:
sudo apt install ffmpeg
# macOS:
brew install ffmpeg
# Windows: https://ffmpeg.org/download.html
```

### 2. Clone & Install

```bash
# Clone the repository
git clone https://github.com/quy92528-del/ai-video-generator.git
cd ai-video-generator

# Create a virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Create directory structure
python setup.py setup_dirs
```

### 3. Configure API Keys

```bash
# Copy the example env file
cp .env.example .env

# Open .env with your favourite editor and add your API keys
nano .env   # or notepad .env on Windows
```

Fill in at least **GOOGLE_GEMINI_API_KEY** to start.  See the [API Key Setup](#api-key-configuration) section for details.

### 4. Run the App

```bash
streamlit run main.py
```

Open **http://localhost:8501** in your browser.

---

## 🔑 API Key Configuration

### Required

| Key | Where to Get | Cost |
|-----|-------------|------|
| `GOOGLE_GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/app/apikey) | Free tier available |

### Recommended (for full functionality)

| Key | Where to Get | Approx Cost |
|-----|-------------|-------------|
| `REPLICATE_API_TOKEN` | [replicate.com/account](https://replicate.com/account/api-tokens) | ~$0.002/image |
| `RUNWAY_API_KEY` | [runwayml.com/account](https://app.runwayml.com/account/api-keys) | ~$0.05/sec |
| `ELEVENLABS_API_KEY` | [elevenlabs.io](https://elevenlabs.io/app/profile/api-key) | ~$0.18/1k chars |

### Optional (Avatar Style)

| Key | Where to Get |
|-----|-------------|
| `HEYGEN_API_KEY` | [heygen.com/settings](https://app.heygen.com/settings/api) |
| `GOOGLE_CLOUD_CREDENTIALS_JSON` | [Google Cloud Console](https://console.cloud.google.com/iam-admin/serviceaccounts) |

---

## 🎬 Video Styles

### 🌊 Motion
Static images animated with AI-generated motion using Runway Gen-3.
- Best for: product showcases, nature content, explainer videos
- Pipeline: Gemini Script → SDXL Image → Runway Image-to-Video → Assembly

### 🎥 Cinematic
Fully AI-generated video scenes from text prompts.
- Best for: storytelling, ads, creative content
- Pipeline: Gemini Script → Runway Text-to-Video → Assembly

### 🧑‍💼 Avatar
AI presenter talking head with automatic lip-sync.
- Best for: news, education, announcements
- Pipeline: Gemini Script → HeyGen Avatar → Wav2Lip Lip-sync → Assembly

---

## 💻 Usage Guide

### Web Interface (Streamlit)

1. **Sidebar** – Set topic, video count (1–500), styles, language, aspect ratio, duration
2. **Generate tab** – Click "▶️ Start Generation" and watch the progress bar
3. **Results tab** – Download individual videos + metadata JSON
4. **Analytics tab** – View cost breakdown and budget usage
5. **Settings page** – Update API keys without editing `.env` manually

### Command Line (Advanced)

```python
from config import get_settings
from brain_module import BrainModule
from visual_engine import VisualEngine
from audio_module import AudioModule
from post_production import PostProduction

settings = get_settings()

# Generate scripts
brain = BrainModule(api_key=settings.google_gemini_api_key)
scripts = brain.generate_batch(
    topic="Top 5 coffee shops in Hanoi",
    count=10,
    styles=["cinematic", "motion"],
    language="vi",
    duration=30,
)

# Build videos
for script in scripts:
    print(script.title)
```

---

## ⚡ Performance

| Config | Videos/Hour | Notes |
|--------|-------------|-------|
| 1 worker, Runway | ~20–30 | Single machine, API limits |
| 10 workers, Runway | ~100–150 | Thread pool, multiple API keys |
| 50 workers, Redis/Celery | ~400–600 | Distributed, requires Redis |

To enable distributed mode:

```bash
# Start Redis
docker run -d -p 6379:6379 redis:alpine

# Start Celery workers
celery -A parallel_processor worker --concurrency=20

# Set in .env
MAX_WORKERS=50
```

---

## 💰 Cost Estimation

With **$300 Google Cloud credit** (and other API keys):

| Style | 30s video | 100 videos | 500 videos |
|-------|-----------|-----------|-----------|
| Motion (SD + Runway) | ~$0.30 | ~$30 | ~$150 |
| Cinematic (Runway only) | ~$0.25 | ~$25 | ~$125 |
| Avatar (HeyGen) | ~$0.15 | ~$15 | ~$75 |

> **Tip**: The app will stop generation automatically when the budget limit is reached.

---

## 🌍 Supported Languages

| Code | Language | TTS Provider |
|------|----------|-------------|
| `vi` | Vietnamese 🇻🇳 | Google TTS / ElevenLabs |
| `en` | English 🇺🇸 | Google TTS / ElevenLabs |
| `zh` | Chinese 🇨🇳 | Google TTS |
| `ja` | Japanese 🇯🇵 | Google TTS |
| `ko` | Korean 🇰🇷 | Google TTS |
| `es` | Spanish 🇪🇸 | Google TTS / ElevenLabs |
| `fr` | French 🇫🇷 | Google TTS / ElevenLabs |

---

## 🔧 Troubleshooting

### "ModuleNotFoundError" on startup
```bash
pip install -r requirements.txt
```

### ffmpeg not found (MoviePy error)
```bash
# Ubuntu: sudo apt install ffmpeg
# macOS: brew install ffmpeg
# Windows: Download from https://ffmpeg.org/download.html and add to PATH
```

### "API key not valid" error
- Double-check your `.env` file for typos
- Make sure you copied the **full** key (no spaces)
- For Gemini, visit [AI Studio](https://aistudio.google.com/app/apikey) to regenerate

### Videos are empty (0 bytes)
- This happens when API keys are missing – the app runs in **mock mode**
- Add valid API keys to `.env` and restart

### Redis connection refused
- Redis is only needed for distributed Celery workers
- For single-machine use, the app falls back to a thread pool automatically

### Out of memory / slow
- Reduce `MAX_WORKERS` in `.env`
- Use `VIDEO_RESOLUTION=720x1280` instead of 1080p

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Google Gemini](https://deepmind.google/technologies/gemini/) for script intelligence
- [Runway ML](https://runwayml.com/) for video generation
- [Replicate](https://replicate.com/) for Stable Diffusion & Wav2Lip
- [ElevenLabs](https://elevenlabs.io/) for voice synthesis
- [HeyGen](https://heygen.com/) for avatar videos
- [Streamlit](https://streamlit.io/) for the web interface
- [MoviePy](https://zulko.github.io/moviepy/) for video editing
