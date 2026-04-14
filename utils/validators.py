"""
utils/validators.py - Input validation, API-key checks, and config validation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger(__name__, file=False)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class ValidationError(ValueError):
    """Raised when validation fails."""


class APIKeyValidationError(ValidationError):
    """Raised when an API key has an unexpected format."""


# ---------------------------------------------------------------------------
# API Key Patterns
# ---------------------------------------------------------------------------

_API_KEY_PATTERNS: Dict[str, re.Pattern[str]] = {
    "google_gemini": re.compile(r"^AIza[0-9A-Za-z\-_]{35}$"),
    "replicate": re.compile(r"^r8_[A-Za-z0-9]{38,}$"),
    "runway": re.compile(r"^[A-Za-z0-9_\-]{20,}$"),
    "elevenlabs": re.compile(r"^[a-f0-9]{32}$"),
}


def validate_api_key(key: str, provider: str) -> bool:
    """Validate the format of an API key for a given provider.

    Args:
        key: The raw API key string.
        provider: Provider name (e.g. ``"google_gemini"``, ``"replicate"``).

    Returns:
        ``True`` when the key matches the expected pattern.

    Raises:
        APIKeyValidationError: When the key is empty or doesn't match.
    """
    if not key or not key.strip():
        raise APIKeyValidationError(f"API key for '{provider}' is empty.")
    pattern = _API_KEY_PATTERNS.get(provider)
    if pattern is None:
        # Unknown provider – just ensure non-empty (already checked above).
        return True
    if not pattern.match(key.strip()):
        raise APIKeyValidationError(
            f"API key for '{provider}' has an unexpected format."
        )
    return True


def validate_topic(topic: str, max_length: int = 500) -> str:
    """Validate and normalise a video topic string.

    Args:
        topic: Raw user input.
        max_length: Maximum allowed character length.

    Returns:
        Stripped topic string.

    Raises:
        ValidationError: On empty or over-long input.
    """
    topic = topic.strip()
    if not topic:
        raise ValidationError("Topic cannot be empty.")
    if len(topic) > max_length:
        raise ValidationError(
            f"Topic exceeds maximum length of {max_length} characters."
        )
    return topic


def validate_video_count(count: Any) -> int:
    """Validate that the requested video count is within [1, 500].

    Args:
        count: Raw value (may be string or int).

    Returns:
        Validated integer count.

    Raises:
        ValidationError: When the value is out of range or not an integer.
    """
    try:
        count = int(count)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Video count must be an integer, got {type(count).__name__}.") from exc
    if count < 1 or count > 500:
        raise ValidationError(f"Video count must be between 1 and 500, got {count}.")
    return count


def validate_duration(seconds: Any, min_s: int = 15, max_s: int = 120) -> int:
    """Validate video duration in seconds.

    Args:
        seconds: Duration value.
        min_s: Minimum seconds allowed.
        max_s: Maximum seconds allowed.

    Returns:
        Validated integer duration.

    Raises:
        ValidationError: When out of range.
    """
    try:
        seconds = int(seconds)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Duration must be an integer number of seconds.") from exc
    if seconds < min_s or seconds > max_s:
        raise ValidationError(
            f"Duration must be between {min_s} and {max_s} seconds, got {seconds}."
        )
    return seconds


def validate_aspect_ratio(ratio: str) -> str:
    """Validate aspect ratio string.

    Args:
        ratio: String like ``"9:16"``.

    Returns:
        Validated ratio string.

    Raises:
        ValidationError: When ratio is not recognised.
    """
    from config import SUPPORTED_ASPECT_RATIOS

    if ratio not in SUPPORTED_ASPECT_RATIOS:
        allowed = list(SUPPORTED_ASPECT_RATIOS.keys())
        raise ValidationError(
            f"Aspect ratio '{ratio}' is not supported.  Allowed: {allowed}"
        )
    return ratio


def validate_language(lang: str) -> str:
    """Validate language code.

    Args:
        lang: ISO language code (e.g. ``"vi"``, ``"en"``).

    Returns:
        Validated language code.

    Raises:
        ValidationError: When language is not supported.
    """
    from config import SUPPORTED_LANGUAGES

    if lang not in SUPPORTED_LANGUAGES:
        allowed = list(SUPPORTED_LANGUAGES.keys())
        raise ValidationError(
            f"Language '{lang}' is not supported.  Allowed: {allowed}"
        )
    return lang


def validate_video_style(style: str) -> str:
    """Validate video style name.

    Args:
        style: Style string (``"motion"``, ``"cinematic"``, or ``"avatar"``).

    Returns:
        Validated style string.

    Raises:
        ValidationError: When style is not recognised.
    """
    from config import SUPPORTED_VIDEO_STYLES

    if style not in SUPPORTED_VIDEO_STYLES:
        raise ValidationError(
            f"Video style '{style}' is not supported.  Allowed: {SUPPORTED_VIDEO_STYLES}"
        )
    return style


def validate_config(settings: Any) -> List[str]:
    """Run all config validation checks and return a list of warnings.

    Args:
        settings: A :class:`config.Settings` instance.

    Returns:
        List of warning strings for optional / missing keys.
    """
    warnings: List[str] = []

    def _warn_if_empty(attr: str, label: str) -> None:
        if not getattr(settings, attr, None):
            warnings.append(f"Optional API key not configured: {label} ({attr})")

    _warn_if_empty("google_gemini_api_key", "Google Gemini")
    _warn_if_empty("runway_api_key", "Runway Gen-3")
    _warn_if_empty("replicate_api_token", "Replicate (Stable Diffusion / Wav2Lip)")
    _warn_if_empty("elevenlabs_api_key", "ElevenLabs TTS")
    _warn_if_empty("heygen_api_key", "HeyGen (Avatar videos)")

    if not warnings:
        log.info("Configuration validated – all API keys present.")
    else:
        for w in warnings:
            log.warning(w)
    return warnings
