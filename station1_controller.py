"""
station1_controller.py - Hybrid Station 1: API Platform Controller

Orchestrates all AI API calls with smart priority, multi-key pooling,
quota management, and automatic fallback to Station 2 (GPU VM) when
all cloud APIs are exhausted or failing.

Flow:
    Trigger → Try Platform APIs (Veo → Grok → Gemini → ...) →
    If all fail → Switch to Station 2 (GPU) → Report result
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import PLATFORM_SCENE_SECONDS
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------

class Platform(str, Enum):
    VEO = "veo"
    GROK = "grok"
    GEMINI = "gemini"
    CHATGPT = "chatgpt"
    GPU = "gpu"  # Station 2 fallback


class TaskType(str, Enum):
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"
    GENERATE_IMAGE = "generate_image"
    GENERATE_AVATAR = "generate_avatar"
    WRITE_SCRIPT = "write_script"
    GENERATE_MUSIC = "generate_music"
    FACE_ANIMATION = "face_animation"


# Default API priority order (tried left to right)
DEFAULT_PRIORITY: List[Platform] = [
    Platform.VEO,
    Platform.GROK,
    Platform.GEMINI,
    Platform.CHATGPT,
]


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class APIKeyEntry:
    """A single API key with usage metadata."""
    key: str
    platform: Platform
    label: str = ""
    quota_exhausted: bool = False
    error_count: int = 0
    last_used: float = 0.0
    expires_at: Optional[str] = None
    account_type: str = ""

    def mark_error(self) -> None:
        self.error_count += 1

    def mark_quota_exhausted(self) -> None:
        self.quota_exhausted = True
        log.warning(
            "API key quota exhausted: platform=%s label=%s", self.platform, self.label
        )

    @property
    def is_available(self) -> bool:
        return not self.quota_exhausted and self.error_count < 5


@dataclass
class PlatformPool:
    """Pool of API keys for one platform with round-robin selection."""

    platform: Platform
    entries: List[APIKeyEntry] = field(default_factory=list)
    _index: int = field(default=0, init=False, repr=False)

    def add_key(
        self,
        key: str,
        label: str = "",
        account_type: str = "",
        expires_at: Optional[str] = None,
    ) -> None:
        self.entries.append(
            APIKeyEntry(
                key=key,
                platform=self.platform,
                label=label,
                account_type=account_type,
                expires_at=expires_at,
            )
        )

    def get_key(self) -> Optional[APIKeyEntry]:
        """Return the next available key (round-robin), or None."""
        available = [e for e in self.entries if e.is_available]
        if not available:
            return None
        entry = available[self._index % len(available)]
        self._index = (self._index + 1) % len(available)
        entry.last_used = time.time()
        return entry

    @property
    def has_keys(self) -> bool:
        return any(e.is_available for e in self.entries)

    @property
    def key_count(self) -> int:
        return sum(1 for e in self.entries if e.is_available)

    def account_info(self) -> List[Dict[str, Any]]:
        """Return summary info for UI display."""
        return [
            {
                "label": e.label or f"Key {i + 1}",
                "account_type": e.account_type,
                "expires_at": e.expires_at or "N/A",
                "available": e.is_available,
                "error_count": e.error_count,
            }
            for i, e in enumerate(self.entries)
        ]


@dataclass
class ControllerResult:
    """Result returned by the Station 1 controller."""

    success: bool
    platform_used: Optional[Platform]
    station_used: int  # 1 = API, 2 = GPU
    output: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    scene_count: int = 0
    total_video_seconds: float = 0.0
    log_entries: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Station 1 Controller
# ---------------------------------------------------------------------------

class Station1Controller:
    """
    Hybrid Station 1: Orchestrates AI platform API calls with priority,
    multi-key pooling, quota management, and fallback to Station 2 (GPU VM).

    Args:
        priority: Ordered list of platforms to try (default: Veo → Grok → Gemini).
        gpu_executor: Optional Station2GPUExecutor instance for fallback.
        status_callback: Optional callable(status_dict) for real-time UI updates.
    """

    def __init__(
        self,
        priority: Optional[List[Platform]] = None,
        gpu_executor: Optional[Any] = None,
        status_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._priority = priority or list(DEFAULT_PRIORITY)
        self._gpu_executor = gpu_executor
        self._status_callback = status_callback
        self._pools: Dict[Platform, PlatformPool] = {p: PlatformPool(p) for p in Platform}
        self._lock = threading.Lock()
        self._log: List[str] = []
        self._running = False

    # ------------------------------------------------------------------ #
    # Key Management                                                        #
    # ------------------------------------------------------------------ #

    def add_api_key(
        self,
        platform: Platform,
        key: str,
        label: str = "",
        account_type: str = "",
        expires_at: Optional[str] = None,
    ) -> None:
        """Register an API key for the given platform."""
        with self._lock:
            self._pools[platform].add_key(
                key=key,
                label=label,
                account_type=account_type,
                expires_at=expires_at,
            )
        self._emit_log(
            f"Added API key for {platform.value} (label={label or 'unnamed'})"
        )

    def get_platform_info(self, platform: Platform) -> Dict[str, Any]:
        """Return account/key summary for a platform (for UI display)."""
        pool = self._pools[platform]
        return {
            "platform": platform.value,
            "key_count": pool.key_count,
            "scene_seconds": PLATFORM_SCENE_SECONDS.get(platform, 0),
            "accounts": pool.account_info(),
        }

    def all_platform_info(self) -> List[Dict[str, Any]]:
        return [self.get_platform_info(p) for p in Platform if p != Platform.GPU]

    # ------------------------------------------------------------------ #
    # Core Dispatch                                                         #
    # ------------------------------------------------------------------ #

    def run_task(
        self,
        task_type: TaskType,
        payload: Dict[str, Any],
        preferred_platforms: Optional[List[Platform]] = None,
        force_gpu: bool = False,
    ) -> ControllerResult:
        """
        Execute a task using the best available platform.

        Priority:
            1. Try each preferred_platforms (or self._priority) in order.
            2. If all API platforms fail → fall back to Station 2 GPU.

        Args:
            task_type: The type of task to perform.
            payload: Task-specific parameters (prompt, image_path, etc.).
            preferred_platforms: Override the default priority order.
            force_gpu: Skip all APIs and go straight to GPU Station 2.

        Returns:
            ControllerResult with result, platform used, and logs.
        """
        self._running = True
        self._log = []
        start = time.time()

        try:
            if force_gpu:
                return self._run_on_gpu(task_type, payload, start)

            platforms = preferred_platforms or self._priority
            last_error = ""

            for platform in platforms:
                if platform == Platform.GPU:
                    continue  # handled separately as fallback

                pool = self._pools[platform]
                if not pool.has_keys:
                    self._emit_log(f"⚠️ No available keys for {platform.value} – skipping.")
                    continue

                key_entry = pool.get_key()
                if key_entry is None:
                    self._emit_log(f"⚠️ All keys exhausted for {platform.value} – skipping.")
                    continue

                self._emit_log(
                    f"🚀 Trying platform: {platform.value} "
                    f"(key={key_entry.label or '(unlabeled)'}) …"
                )
                try:
                    output = self._dispatch_to_platform(
                        platform=platform,
                        key_entry=key_entry,
                        task_type=task_type,
                        payload=payload,
                    )
                    scene_count = payload.get("scene_count", 1)
                    scene_secs = PLATFORM_SCENE_SECONDS.get(platform, 0)
                    total_secs = scene_count * scene_secs

                    self._emit_log(
                        f"✅ Completed on {platform.value}: "
                        f"{scene_count} scenes × {scene_secs}s = {total_secs}s total."
                    )
                    return ControllerResult(
                        success=True,
                        platform_used=platform,
                        station_used=1,
                        output=output,
                        duration_seconds=time.time() - start,
                        scene_count=scene_count,
                        total_video_seconds=total_secs,
                        log_entries=list(self._log),
                    )

                except QuotaExceededError:
                    key_entry.mark_quota_exhausted()
                    self._emit_log(
                        f"🔴 Quota exceeded on {platform.value}. Trying next platform…"
                    )
                    last_error = f"Quota exceeded: {platform.value}"

                except APICallError as exc:
                    key_entry.mark_error()
                    self._emit_log(
                        f"🔴 API error on {platform.value}: {exc}. Trying next platform…"
                    )
                    last_error = str(exc)

            # All APIs failed or exhausted — fall back to GPU Station 2
            self._emit_log(
                "⚡ All API platforms failed or exhausted — switching to Station 2 (GPU VM)…"
            )
            return self._run_on_gpu(task_type, payload, start, prior_error=last_error)

        except Exception as exc:  # pylint: disable=broad-except
            self._emit_log(f"💥 Unexpected controller error: {exc}")
            return ControllerResult(
                success=False,
                platform_used=None,
                station_used=1,
                error=str(exc),
                duration_seconds=time.time() - start,
                log_entries=list(self._log),
            )
        finally:
            self._running = False

    # ------------------------------------------------------------------ #
    # Platform Dispatcher                                                   #
    # ------------------------------------------------------------------ #

    def _dispatch_to_platform(
        self,
        platform: Platform,
        key_entry: APIKeyEntry,
        task_type: TaskType,
        payload: Dict[str, Any],
    ) -> Any:
        """Route a task to the appropriate platform handler."""
        handler_map: Dict[Platform, Callable[..., Any]] = {
            Platform.VEO: self._call_veo,
            Platform.GROK: self._call_grok,
            Platform.GEMINI: self._call_gemini,
            Platform.CHATGPT: self._call_chatgpt,
        }
        handler = handler_map.get(platform)
        if handler is None:
            raise APICallError(f"No handler for platform: {platform.value}")
        return handler(key=key_entry.key, task_type=task_type, payload=payload)

    # ------------------------------------------------------------------ #
    # Platform Handlers (TODO: fill in real API calls per platform)         #
    # ------------------------------------------------------------------ #

    def _call_veo(self, key: str, task_type: TaskType, payload: Dict[str, Any]) -> Any:
        """
        Call Google Veo API for video generation.
        TODO: Implement real Veo API call using google-generativeai or REST.
        """
        # Example payload keys: prompt, image_path, duration_seconds, aspect_ratio
        log.debug("Veo API call: task=%s payload_keys=%s", task_type, list(payload.keys()))

        if task_type in (TaskType.TEXT_TO_VIDEO, TaskType.IMAGE_TO_VIDEO):
            return self._veo_generate_video(key=key, payload=payload)
        raise APICallError(f"Veo does not support task type: {task_type}")

    def _veo_generate_video(self, key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        TODO: Replace stub with real Veo REST / SDK call.

        Expected payload:
            prompt (str): Video description.
            image_path (str, optional): Source image for image-to-video.
            duration_seconds (float): Desired duration (Veo produces 8 s clips).
            aspect_ratio (str): e.g. "16:9", "9:16".
            model (str): e.g. "veo-3", "veo-3-fast".
        """
        log.info(
            "[Veo] Generating video: prompt='%s' duration=%s",
            payload.get("prompt", "")[:60],
            payload.get("duration_seconds", 8),
        )
        # Stub response — replace with actual API response parsing
        return {
            "provider": "veo",
            "video_url": None,
            "duration_seconds": payload.get("duration_seconds", 8),
            "status": "stub",
        }

    def _call_grok(self, key: str, task_type: TaskType, payload: Dict[str, Any]) -> Any:
        """
        Call xAI Grok API for video/image generation.
        TODO: Implement real Grok API call.
        Grok produces 6 s video clips; supports extend-video for continuity.
        """
        log.debug("Grok API call: task=%s", task_type)

        if task_type == TaskType.WRITE_SCRIPT:
            return self._grok_write_script(key=key, payload=payload)
        if task_type in (TaskType.TEXT_TO_VIDEO, TaskType.IMAGE_TO_VIDEO):
            return self._grok_generate_video(key=key, payload=payload)
        raise APICallError(f"Grok does not support task type: {task_type}")

    def _grok_write_script(self, key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        TODO: Replace stub with real Grok chat completion call.
        """
        log.info("[Grok] Writing script for topic='%s'", payload.get("topic", "")[:60])
        return {"provider": "grok", "script": None, "status": "stub"}

    def _grok_generate_video(self, key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        TODO: Replace stub with real Grok video generation call.

        Grok produces 6 s clips.  Use 'extend_from_video_id' in payload
        to chain scenes for narrative continuity.
        """
        log.info(
            "[Grok] Generating video clip: prompt='%s'",
            payload.get("prompt", "")[:60],
        )
        return {
            "provider": "grok",
            "video_url": None,
            "duration_seconds": 6,
            "extend_video_id": None,
            "status": "stub",
        }

    def _call_gemini(self, key: str, task_type: TaskType, payload: Dict[str, Any]) -> Any:
        """
        Call Google Gemini API for script writing, image generation, etc.
        TODO: Implement using google-generativeai SDK.
        """
        log.debug("Gemini API call: task=%s", task_type)
        if task_type == TaskType.WRITE_SCRIPT:
            return self._gemini_write_script(key=key, payload=payload)
        if task_type == TaskType.GENERATE_IMAGE:
            return self._gemini_generate_image(key=key, payload=payload)
        raise APICallError(f"Gemini does not support task type: {task_type}")

    def _gemini_write_script(self, key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        TODO: Replace stub with real Gemini generateContent call.
        """
        log.info("[Gemini] Writing script for topic='%s'", payload.get("topic", "")[:60])
        return {"provider": "gemini", "script": None, "status": "stub"}

    def _gemini_generate_image(self, key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        TODO: Replace stub with real Gemini Imagen / Imagen 3 call.
        """
        log.info("[Gemini] Generating image: prompt='%s'", payload.get("prompt", "")[:60])
        return {"provider": "gemini", "image_path": None, "status": "stub"}

    def _call_chatgpt(self, key: str, task_type: TaskType, payload: Dict[str, Any]) -> Any:
        """
        Call OpenAI ChatGPT API for script / prompt writing.
        TODO: Implement using openai SDK.
        """
        log.debug("ChatGPT API call: task=%s", task_type)
        if task_type == TaskType.WRITE_SCRIPT:
            return self._chatgpt_write_script(key=key, payload=payload)
        raise APICallError(f"ChatGPT does not support task type: {task_type}")

    def _chatgpt_write_script(self, key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        TODO: Replace stub with real OpenAI chat completions call.
        """
        log.info("[ChatGPT] Writing script for topic='%s'", payload.get("topic", "")[:60])
        return {"provider": "chatgpt", "script": None, "status": "stub"}

    # ------------------------------------------------------------------ #
    # Station 2 GPU Fallback                                               #
    # ------------------------------------------------------------------ #

    def _run_on_gpu(
        self,
        task_type: TaskType,
        payload: Dict[str, Any],
        start: float,
        prior_error: str = "",
    ) -> ControllerResult:
        """Delegate the task to Station 2 GPU executor with safe shutdown."""
        if self._gpu_executor is None:
            msg = "Station 2 GPU executor not configured."
            self._emit_log(f"❌ {msg}")
            return ControllerResult(
                success=False,
                platform_used=Platform.GPU,
                station_used=2,
                error=msg,
                duration_seconds=time.time() - start,
                log_entries=list(self._log),
            )

        self._emit_log("🖥️ Delegating to Station 2 (GPU VM) …")
        try:
            gpu_result = self._gpu_executor.execute(
                task_type=task_type.value,
                payload=payload,
            )
            self._emit_log(
                f"✅ Station 2 completed: success={gpu_result.get('success')}"
            )
            return ControllerResult(
                success=gpu_result.get("success", False),
                platform_used=Platform.GPU,
                station_used=2,
                output=gpu_result.get("output"),
                error=gpu_result.get("error"),
                duration_seconds=time.time() - start,
                log_entries=list(self._log),
            )
        except Exception as exc:  # pylint: disable=broad-except
            self._emit_log(f"❌ Station 2 failed: {exc}")
            return ControllerResult(
                success=False,
                platform_used=Platform.GPU,
                station_used=2,
                error=str(exc),
                duration_seconds=time.time() - start,
                log_entries=list(self._log),
            )

    # ------------------------------------------------------------------ #
    # Scene Duration Helpers                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_total_duration(
        platform: Platform, scene_count: int
    ) -> Tuple[float, str]:
        """
        Calculate total video duration for the given platform and scene count.

        Returns:
            (total_seconds, human_readable_label)
        """
        secs_per_scene = PLATFORM_SCENE_SECONDS.get(platform, 8.0)
        total = scene_count * secs_per_scene
        label = f"~{total:.0f}s ({scene_count} scenes × {secs_per_scene:.0f}s)"
        return total, label

    # ------------------------------------------------------------------ #
    # Logging / Status                                                     #
    # ------------------------------------------------------------------ #

    def _emit_log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self._log.append(entry)
        log.info(msg)
        if self._status_callback:
            try:
                self._status_callback({"log": entry, "running": self._running})
            except Exception:  # pylint: disable=broad-except
                pass

    @property
    def log_entries(self) -> List[str]:
        return list(self._log)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class APICallError(Exception):
    """Raised when an API call fails (non-quota reason)."""


class QuotaExceededError(APICallError):
    """Raised when an API key's quota is exhausted."""
