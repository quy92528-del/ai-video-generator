"""
brain_module.py - Script & metadata generation using Google Gemini.

Generates video scripts, scene descriptions, hashtags, and titles for
each requested topic + style combination.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.logger import get_logger
from utils.helpers import content_hash

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    """A single scene within a video script."""
    index: int
    description: str          # Visual description for image/video generation
    narration: str            # Voice-over / narration text
    duration_seconds: int = 5
    keywords: List[str] = field(default_factory=list)
    mood: str = "neutral"
    camera_angle: str = "medium shot"


@dataclass
class VideoScript:
    """Complete script for one video."""
    topic: str
    title: str
    description: str
    style: str
    language: str
    duration_seconds: int
    scenes: List[Scene]
    tags: List[str] = field(default_factory=list)
    cta: str = ""             # Call-to-action text
    background_music_mood: str = "upbeat"
    cache_key: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "title": self.title,
            "description": self.description,
            "style": self.style,
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "scenes": [
                {
                    "index": s.index,
                    "description": s.description,
                    "narration": s.narration,
                    "duration_seconds": s.duration_seconds,
                    "keywords": s.keywords,
                    "mood": s.mood,
                    "camera_angle": s.camera_angle,
                }
                for s in self.scenes
            ],
            "tags": self.tags,
            "cta": self.cta,
            "background_music_mood": self.background_music_mood,
        }


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SCRIPT_PROMPT_TEMPLATE = """
You are an expert viral video scriptwriter. Generate a complete video script in JSON format.

**Topic:** {topic}
**Style:** {style}
**Language:** {language}
**Duration:** {duration} seconds
**Number of scenes:** {scene_count}

Return ONLY valid JSON with this exact structure:
{{
  "title": "Engaging title (max 100 chars)",
  "description": "SEO-optimised description (max 500 chars)",
  "tags": ["tag1", "tag2", ...],
  "cta": "Call-to-action text",
  "background_music_mood": "upbeat|dramatic|calm|emotional|energetic",
  "scenes": [
    {{
      "index": 0,
      "description": "Detailed visual description for image generation",
      "narration": "Voice-over text in {language}",
      "duration_seconds": 5,
      "keywords": ["keyword1", "keyword2"],
      "mood": "neutral|happy|dramatic|mysterious|energetic",
      "camera_angle": "close-up|medium shot|wide shot|aerial"
    }}
  ]
}}

Rules:
- All narration text must be in {language}.
- Scene descriptions must be in English for image generation.
- Total scene durations must sum to approximately {duration} seconds.
- Make content engaging and viral-worthy.
- Include {scene_count} scenes.
""".strip()


# ---------------------------------------------------------------------------
# BrainModule
# ---------------------------------------------------------------------------

class BrainModule:
    """Generates video scripts using Google Gemini.

    Args:
        api_key: Google Gemini API key.
        model: Gemini model name (default: ``"gemini-1.5-pro"``).
        cache: Optional dict-like cache for storing generated scripts.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-pro",
        cache: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._cache: Dict[str, Any] = cache if cache is not None else {}
        self._client: Any = None
        self._init_client()

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def _init_client(self) -> None:
        """Initialise the Gemini client (lazy-imports to avoid hard dep)."""
        try:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=self._api_key)
            self._client = genai.GenerativeModel(self._model)
            log.info("Gemini client initialised (model=%s).", self._model)
        except ImportError:
            log.warning(
                "google-generativeai not installed – BrainModule in mock mode."
            )
        except Exception as exc:  # pylint: disable=broad-except
            log.error("Failed to initialise Gemini client: %s", exc)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def generate_script(
        self,
        topic: str,
        style: str = "cinematic",
        language: str = "vi",
        duration: int = 30,
        variation_index: int = 0,
    ) -> VideoScript:
        """Generate a video script for *topic*.

        Args:
            topic: Video topic / description.
            style: One of ``"motion"``, ``"cinematic"``, ``"avatar"``.
            language: ISO language code (e.g. ``"vi"``, ``"en"``).
            duration: Target video duration in seconds.
            variation_index: When generating multiple variations, pass 1, 2, …
                             to get different outputs for the same topic.

        Returns:
            A :class:`VideoScript` instance.
        """
        cache_key = content_hash(
            f"{topic}|{style}|{language}|{duration}|{variation_index}"
        )
        if cache_key in self._cache:
            log.debug("Cache hit for key %s", cache_key)
            return self._cache[cache_key]

        scene_count = max(3, duration // 10)
        prompt = _SCRIPT_PROMPT_TEMPLATE.format(
            topic=topic,
            style=style,
            language=language,
            duration=duration,
            scene_count=scene_count,
        )
        if variation_index > 0:
            prompt += f"\n\nVariation {variation_index}: provide a distinctly different angle."

        raw = self._call_gemini(prompt)
        script = self._parse_script(raw, topic, style, language, duration, cache_key)
        self._cache[cache_key] = script
        return script

    def generate_batch(
        self,
        topic: str,
        count: int,
        styles: Optional[List[str]] = None,
        language: str = "vi",
        duration: int = 30,
    ) -> List[VideoScript]:
        """Generate *count* script variations for a topic.

        Args:
            topic: Video topic.
            count: Number of scripts to generate.
            styles: List of styles to cycle through.
            language: Language code.
            duration: Duration in seconds.

        Returns:
            List of :class:`VideoScript` objects.
        """
        if not styles:
            styles = ["cinematic", "motion", "avatar"]
        scripts: List[VideoScript] = []
        for i in range(count):
            style = styles[i % len(styles)]
            try:
                script = self.generate_script(
                    topic=topic,
                    style=style,
                    language=language,
                    duration=duration,
                    variation_index=i,
                )
                scripts.append(script)
                log.info("Generated script %d/%d (style=%s).", i + 1, count, style)
            except Exception as exc:  # pylint: disable=broad-except
                log.error("Script generation failed for variation %d: %s", i, exc)
                scripts.append(
                    self._fallback_script(topic, style, language, duration, i)
                )
            # Respect rate limits – small sleep between API calls.
            if i < count - 1:
                time.sleep(0.3)
        return scripts

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _call_gemini(self, prompt: str) -> str:
        """Call the Gemini API and return the text response."""
        if self._client is None:
            log.warning("Gemini client unavailable – returning mock response.")
            return self._mock_response(prompt)
        try:
            response = self._client.generate_content(prompt)
            return response.text
        except Exception as exc:  # pylint: disable=broad-except
            log.error("Gemini API error: %s", exc)
            return self._mock_response(prompt)

    def _parse_script(
        self,
        raw: str,
        topic: str,
        style: str,
        language: str,
        duration: int,
        cache_key: str,
    ) -> VideoScript:
        """Parse Gemini response JSON into a :class:`VideoScript`."""
        # Strip markdown code fences if present.
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"```\s*$", "", raw)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("Could not parse Gemini JSON – using fallback script.")
            return self._fallback_script(topic, style, language, duration, 0)

        scenes = [
            Scene(
                index=s.get("index", i),
                description=s.get("description", f"Scene {i}"),
                narration=s.get("narration", ""),
                duration_seconds=s.get("duration_seconds", 5),
                keywords=s.get("keywords", []),
                mood=s.get("mood", "neutral"),
                camera_angle=s.get("camera_angle", "medium shot"),
            )
            for i, s in enumerate(data.get("scenes", []))
        ]
        return VideoScript(
            topic=topic,
            title=data.get("title", topic),
            description=data.get("description", ""),
            style=style,
            language=language,
            duration_seconds=duration,
            scenes=scenes,
            tags=data.get("tags", []),
            cta=data.get("cta", ""),
            background_music_mood=data.get("background_music_mood", "upbeat"),
            cache_key=cache_key,
        )

    @staticmethod
    def _fallback_script(
        topic: str, style: str, language: str, duration: int, index: int
    ) -> VideoScript:
        """Return a minimal fallback script when Gemini is unavailable."""
        scene_count = max(3, duration // 10)
        scenes = [
            Scene(
                index=i,
                description=f"Visual representation of {topic}, scene {i + 1}",
                narration=f"{topic} - part {i + 1}",
                duration_seconds=duration // scene_count,
                keywords=[topic],
                mood="neutral",
                camera_angle="medium shot",
            )
            for i in range(scene_count)
        ]
        return VideoScript(
            topic=topic,
            title=f"{topic} (variation {index + 1})",
            description=f"AI-generated video about {topic}.",
            style=style,
            language=language,
            duration_seconds=duration,
            scenes=scenes,
            tags=[topic, style],
            cta="Like and subscribe!",
            background_music_mood="upbeat",
            cache_key="",
        )

    @staticmethod
    def _mock_response(prompt: str) -> str:  # noqa: ARG004
        """Return a hard-coded mock JSON when Gemini is unavailable."""
        return json.dumps(
            {
                "title": "AI Generated Video",
                "description": "An amazing AI-generated video.",
                "tags": ["ai", "video", "automation"],
                "cta": "Subscribe for more!",
                "background_music_mood": "upbeat",
                "scenes": [
                    {
                        "index": i,
                        "description": f"A stunning visual scene {i + 1}",
                        "narration": f"Scene {i + 1} narration text.",
                        "duration_seconds": 10,
                        "keywords": ["visual", "stunning"],
                        "mood": "neutral",
                        "camera_angle": "medium shot",
                    }
                    for i in range(3)
                ],
            }
        )


# ---------------------------------------------------------------------------
# Lazy import guard
# ---------------------------------------------------------------------------
import re  # noqa: E402 – needed by _parse_script
