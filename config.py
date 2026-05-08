"""
config.py - Configuration & API Settings for AI Video Generator Suite
Manages all environment variables, API keys, and system configuration.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator

try:
    from pydantic_settings import BaseSettings  # Pydantic v2
except ImportError:
    from pydantic import BaseSettings  # type: ignore[no-redef]  # Pydantic v1 fallback

# ---------------------------------------------------------------------------
# Load .env file from project root
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
load_dotenv(_ROOT / ".env", override=False)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class ConfigurationError(Exception):
    """Raised when a required configuration value is missing or invalid."""


class APIKeyError(ConfigurationError):
    """Raised when an API key is missing or malformed."""


# ---------------------------------------------------------------------------
# Settings Model (validated with Pydantic)
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Central settings object.  All values can be overridden via env vars."""

    # ---- Google APIs -------------------------------------------------------
    google_gemini_api_key: str = Field(default="", alias="GOOGLE_GEMINI_API_KEY")
    google_cloud_project_id: str = Field(default="", alias="GOOGLE_CLOUD_PROJECT_ID")
    google_cloud_credentials_json: str = Field(
        default="", alias="GOOGLE_CLOUD_CREDENTIALS_JSON"
    )
    google_cloud_location: str = Field(
        default="us-central1", alias="GOOGLE_CLOUD_LOCATION"
    )

    # ---- Runway API --------------------------------------------------------
    runway_api_key: str = Field(default="", alias="RUNWAY_API_KEY")
    runway_api_version: str = Field(default="2024-11-06", alias="RUNWAY_API_VERSION")

    # ---- Replicate (Stable Diffusion, Wav2Lip) ----------------------------
    replicate_api_token: str = Field(default="", alias="REPLICATE_API_TOKEN")

    # ---- ElevenLabs --------------------------------------------------------
    elevenlabs_api_key: str = Field(default="", alias="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str = Field(
        default="21m00Tcm4TlvDq8ikWAM", alias="ELEVENLABS_VOICE_ID"
    )  # Rachel (default)

    # ---- HeyGen / D-ID (Avatar videos) ------------------------------------
    heygen_api_key: str = Field(default="", alias="HEYGEN_API_KEY")
    did_api_key: str = Field(default="", alias="DID_API_KEY")

    # ---- Veo (Google) ------------------------------------------------------
    veo_api_key: str = Field(default="", alias="VEO_API_KEY")
    veo_api_keys_extra: str = Field(
        default="", alias="VEO_API_KEYS_EXTRA"
    )  # Comma-separated additional Veo API keys
    veo_model: str = Field(default="veo-3", alias="VEO_MODEL")
    veo_account_type: str = Field(default="", alias="VEO_ACCOUNT_TYPE")
    veo_expires_at: str = Field(default="", alias="VEO_EXPIRES_AT")

    # ---- Grok (xAI) --------------------------------------------------------
    grok_api_key: str = Field(default="", alias="GROK_API_KEY")
    grok_api_keys_extra: str = Field(
        default="", alias="GROK_API_KEYS_EXTRA"
    )  # Comma-separated additional Grok API keys
    grok_model: str = Field(default="grok-2-vision-1212", alias="GROK_MODEL")
    grok_account_type: str = Field(default="", alias="GROK_ACCOUNT_TYPE")
    grok_expires_at: str = Field(default="", alias="GROK_EXPIRES_AT")

    # ---- ChatGPT (OpenAI) --------------------------------------------------
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_api_keys_extra: str = Field(default="", alias="OPENAI_API_KEYS_EXTRA")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")

    # ---- Hybrid Station 2 - GCP GPU VM ------------------------------------
    gcp_project_id: str = Field(default="", alias="GCP_PROJECT_ID")
    gcp_zone: str = Field(default="us-central1-a", alias="GCP_ZONE")
    gcp_gpu_instance_name: str = Field(
        default="ai-video-gpu-worker", alias="GCP_GPU_INSTANCE_NAME"
    )
    gcp_ssh_user: str = Field(default="ubuntu", alias="GCP_SSH_USER")
    gcp_worker_script: str = Field(
        default="/opt/ai-video/worker.py", alias="GCP_WORKER_SCRIPT"
    )
    gcp_vm_startup_timeout: int = Field(
        default=300, alias="GCP_VM_STARTUP_TIMEOUT"
    )  # seconds
    gcp_vm_task_timeout: int = Field(
        default=3600, alias="GCP_VM_TASK_TIMEOUT"
    )  # seconds
    enable_station2_gpu: bool = Field(
        default=False, alias="ENABLE_STATION2_GPU"
    )  # Require explicit opt-in

    # ---- Platform API Priority (comma-separated, e.g. "veo,grok,gemini") --
    api_priority: str = Field(
        default="veo,grok,gemini,chatgpt", alias="API_PRIORITY"
    )

    # ---- Redis / Celery ----------------------------------------------------
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0", alias="CELERY_BROKER_URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/1", alias="CELERY_RESULT_BACKEND"
    )

    # ---- Processing --------------------------------------------------------
    max_workers: int = Field(default=10, alias="MAX_WORKERS")
    batch_size: int = Field(default=50, alias="BATCH_SIZE")
    request_timeout: int = Field(default=120, alias="REQUEST_TIMEOUT")
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    retry_delay: float = Field(default=2.0, alias="RETRY_DELAY")

    # ---- Video Defaults ----------------------------------------------------
    default_aspect_ratio: str = Field(default="9:16", alias="DEFAULT_ASPECT_RATIO")
    default_duration: int = Field(default=30, alias="DEFAULT_DURATION")  # seconds
    default_language: str = Field(default="vi", alias="DEFAULT_LANGUAGE")
    default_video_style: str = Field(default="cinematic", alias="DEFAULT_VIDEO_STYLE")
    video_fps: int = Field(default=30, alias="VIDEO_FPS")
    video_resolution: str = Field(default="1080x1920", alias="VIDEO_RESOLUTION")

    # ---- Budget ------------------------------------------------------------
    budget_limit_usd: float = Field(default=300.0, alias="BUDGET_LIMIT_USD")
    cost_alert_threshold: float = Field(
        default=0.8, alias="COST_ALERT_THRESHOLD"
    )  # 80 % of budget

    # ---- Storage -----------------------------------------------------------
    output_dir: str = Field(default="output", alias="OUTPUT_DIR")
    assets_dir: str = Field(default="assets", alias="ASSETS_DIR")
    cache_dir: str = Field(default=".cache", alias="CACHE_DIR")
    log_dir: str = Field(default="output/logs", alias="LOG_DIR")

    # ---- Feature Flags -----------------------------------------------------
    enable_face_consistency: bool = Field(
        default=True, alias="ENABLE_FACE_CONSISTENCY"
    )
    enable_audio_ducking: bool = Field(default=True, alias="ENABLE_AUDIO_DUCKING")
    enable_cost_tracking: bool = Field(default=True, alias="ENABLE_COST_TRACKING")
    enable_caching: bool = Field(default=True, alias="ENABLE_CACHING")
    debug_mode: bool = Field(default=False, alias="DEBUG_MODE")

    # ---- Model Names -------------------------------------------------------
    gemini_model: str = Field(
        default="gemini-1.5-pro", alias="GEMINI_MODEL"
    )
    stable_diffusion_model: str = Field(
        default="stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
        alias="STABLE_DIFFUSION_MODEL",
    )
    wav2lip_model: str = Field(
        default="devxpy/cog-wav2lip:8d65e3f4f42985ff6a9e54fd5e48a23dc32a7d0a",
        alias="WAV2LIP_MODEL",
    )

    model_config = {"env_file": ".env", "populate_by_name": True}

    @field_validator("max_workers")
    @classmethod
    def _clamp_workers(cls, v: int) -> int:
        return max(1, min(v, 100))

    @field_validator("budget_limit_usd")
    @classmethod
    def _positive_budget(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Budget limit must be positive.")
        return v


# ---------------------------------------------------------------------------
# API Cost Constants (USD)
# ---------------------------------------------------------------------------

API_COSTS = {
    # Gemini
    "gemini_input_per_1k_tokens": 0.00025,
    "gemini_output_per_1k_tokens": 0.0005,
    # Stable Diffusion (Replicate)
    "stable_diffusion_per_image": 0.0023,
    # Runway Gen-3
    "runway_per_second": 0.05,
    "runway_min_seconds": 5,
    # ElevenLabs TTS
    "elevenlabs_per_1k_chars": 0.18,
    # Google Cloud TTS
    "gcloud_tts_per_1m_chars": 16.0,
    # Wav2Lip (Replicate)
    "wav2lip_per_run": 0.0575,
    # HeyGen
    "heygen_per_minute": 0.10,
    # D-ID
    "did_per_second": 0.007,
    # Veo
    "veo_per_second": 0.12,
    # Grok video
    "grok_video_per_clip": 0.05,
    # ChatGPT / OpenAI
    "openai_input_per_1k_tokens": 0.005,
    "openai_output_per_1k_tokens": 0.015,
}

# ---------------------------------------------------------------------------
# Per-platform scene duration (seconds per generation call)
# ---------------------------------------------------------------------------

PLATFORM_SCENE_SECONDS = {
    "veo": 8.0,   # Veo 3 produces 8 s per generation
    "grok": 6.0,  # Grok produces 6 s per generation
    "runway": 5.0,
    "heygen": 30.0,
}

# ---------------------------------------------------------------------------
# Supported Values
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES = {
    "vi": "Vietnamese",
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
}

SUPPORTED_ASPECT_RATIOS = {
    "9:16": (1080, 1920),    # Vertical / Reels / TikTok
    "16:9": (1920, 1080),    # Horizontal / YouTube
    "1:1": (1080, 1080),     # Square / Instagram
    "4:3": (1440, 1080),     # Classic TV / Presentations
    "3:4": (1080, 1440),     # Tall portrait
    "21:9": (2560, 1080),    # Ultra-wide / Cinematic
    "6:19": (720, 2280),     # Tall mobile (user-requested)
    "19:6": (2280, 720),     # Wide mobile (user-requested)
}

SUPPORTED_VIDEO_STYLES = [
    "motion",
    "cinematic",
    "avatar",
    "animated",
    "3d",
    "pixar",
    "anime",
    "realistic",
    "documentary",
    "music_video",
]

SUPPORTED_PLATFORMS = ["veo", "grok", "gemini", "chatgpt", "runway", "heygen"]

SUPPORTED_VIDEO_DURATIONS = list(range(15, 121, 15))  # 15 s intervals up to 120 s

# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _ensure_dirs(settings: Settings) -> None:
    """Create output / cache / log directories if they don't exist."""
    dirs = [
        settings.output_dir,
        f"{settings.output_dir}/videos",
        f"{settings.output_dir}/thumbnails",
        f"{settings.output_dir}/metadata",
        f"{settings.output_dir}/logs",
        settings.assets_dir,
        f"{settings.assets_dir}/music",
        f"{settings.assets_dir}/sounds",
        f"{settings.assets_dir}/fonts",
        f"{settings.assets_dir}/templates",
        settings.cache_dir,
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance and ensure directories exist."""
    s = Settings()
    _ensure_dirs(s)
    return s
