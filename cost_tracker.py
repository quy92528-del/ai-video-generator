"""
cost_tracker.py - Real-time cost calculation, budget monitoring, and
per-API usage logging.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger
from utils.helpers import ensure_dir, write_json

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class APIUsageRecord:
    """Represents a single API call and its cost."""
    timestamp: float
    provider: str
    operation: str
    units: float          # tokens, seconds, characters, images, etc.
    unit_name: str        # "tokens", "seconds", "chars", "images"
    cost_usd: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "provider": self.provider,
            "operation": self.operation,
            "units": self.units,
            "unit_name": self.unit_name,
            "cost_usd": round(self.cost_usd, 6),
            "metadata": self.metadata,
        }


@dataclass
class CostSummary:
    """Aggregated cost report."""
    total_usd: float
    budget_usd: float
    remaining_usd: float
    usage_pct: float
    by_provider: Dict[str, float]
    by_operation: Dict[str, float]
    record_count: int
    videos_generated: int
    cost_per_video: float


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

class CostTracker:
    """Thread-safe cost tracker with budget monitoring.

    Args:
        budget_usd: Total budget limit in USD.
        alert_threshold: Fraction of budget that triggers a warning (0–1).
        log_dir: Directory for persisting cost logs.
        costs: Optional custom cost table (overrides defaults).
    """

    _DEFAULT_COSTS: Dict[str, Dict[str, float]] = {
        "gemini": {
            "input_per_1k_tokens": 0.00025,
            "output_per_1k_tokens": 0.0005,
        },
        "stable_diffusion": {
            "per_image": 0.0023,
        },
        "runway": {
            "per_second": 0.05,
        },
        "elevenlabs": {
            "per_1k_chars": 0.18,
        },
        "google_tts": {
            "per_1m_chars": 16.0,
        },
        "wav2lip": {
            "per_run": 0.0575,
        },
        "heygen": {
            "per_minute": 0.10,
        },
        "did": {
            "per_second": 0.007,
        },
    }

    def __init__(
        self,
        budget_usd: float = 300.0,
        alert_threshold: float = 0.8,
        log_dir: str = "output/logs",
        costs: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self._budget = budget_usd
        self._alert_threshold = alert_threshold
        self._log_dir = Path(log_dir)
        self._costs = costs or dict(self._DEFAULT_COSTS)
        self._records: List[APIUsageRecord] = []
        self._videos_generated: int = 0
        self._lock = threading.Lock()
        self._alert_sent = False
        ensure_dir(self._log_dir)

    # ------------------------------------------------------------------ #
    # Recording Usage                                                      #
    # ------------------------------------------------------------------ #

    def record_gemini(
        self,
        input_tokens: int,
        output_tokens: int,
        operation: str = "generate_content",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Record a Gemini API call cost.

        Args:
            input_tokens: Number of input tokens consumed.
            output_tokens: Number of output tokens generated.
            operation: Descriptive operation name.
            metadata: Optional extra metadata.

        Returns:
            Estimated cost in USD.
        """
        costs = self._costs["gemini"]
        cost = (
            input_tokens / 1000.0 * costs["input_per_1k_tokens"]
            + output_tokens / 1000.0 * costs["output_per_1k_tokens"]
        )
        return self._record(
            provider="gemini",
            operation=operation,
            units=input_tokens + output_tokens,
            unit_name="tokens",
            cost=cost,
            metadata=metadata or {},
        )

    def record_image_generation(
        self,
        count: int = 1,
        provider: str = "stable_diffusion",
        operation: str = "generate_image",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Record an image generation cost.

        Args:
            count: Number of images generated.
            provider: Provider name (``"stable_diffusion"``).
            operation: Operation label.
            metadata: Optional extra metadata.

        Returns:
            Estimated cost in USD.
        """
        per_image = self._costs.get(provider, {}).get("per_image", 0.0023)
        cost = count * per_image
        return self._record(
            provider=provider,
            operation=operation,
            units=float(count),
            unit_name="images",
            cost=cost,
            metadata=metadata or {},
        )

    def record_video_generation(
        self,
        seconds: float,
        provider: str = "runway",
        operation: str = "image_to_video",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Record a video generation cost.

        Args:
            seconds: Duration of generated video in seconds.
            provider: Provider (``"runway"``, ``"heygen"``, ``"did"``).
            operation: Operation label.
            metadata: Optional extra metadata.

        Returns:
            Estimated cost in USD.
        """
        p = self._costs.get(provider, {})
        if provider == "runway":
            cost = seconds * p.get("per_second", 0.05)
        elif provider == "heygen":
            cost = (seconds / 60.0) * p.get("per_minute", 0.10)
        elif provider == "did":
            cost = seconds * p.get("per_second", 0.007)
        else:
            cost = seconds * 0.05
        return self._record(
            provider=provider,
            operation=operation,
            units=seconds,
            unit_name="seconds",
            cost=cost,
            metadata=metadata or {},
        )

    def record_tts(
        self,
        characters: int,
        provider: str = "google_tts",
        operation: str = "synthesize",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Record a TTS synthesis cost.

        Args:
            characters: Number of characters synthesised.
            provider: TTS provider (``"google_tts"`` or ``"elevenlabs"``).
            operation: Operation label.
            metadata: Optional extra metadata.

        Returns:
            Estimated cost in USD.
        """
        p = self._costs.get(provider, {})
        if provider == "google_tts":
            cost = characters / 1_000_000 * p.get("per_1m_chars", 16.0)
        elif provider == "elevenlabs":
            cost = characters / 1000.0 * p.get("per_1k_chars", 0.18)
        else:
            cost = 0.0
        return self._record(
            provider=provider,
            operation=operation,
            units=float(characters),
            unit_name="chars",
            cost=cost,
            metadata=metadata or {},
        )

    def record_wav2lip(
        self, metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """Record a Wav2Lip run cost."""
        cost = self._costs["wav2lip"]["per_run"]
        return self._record(
            provider="replicate",
            operation="wav2lip",
            units=1.0,
            unit_name="runs",
            cost=cost,
            metadata=metadata or {},
        )

    def increment_videos(self, count: int = 1) -> None:
        """Increment the counter of completed videos."""
        with self._lock:
            self._videos_generated += count

    # ------------------------------------------------------------------ #
    # Budget Queries                                                       #
    # ------------------------------------------------------------------ #

    @property
    def total_cost(self) -> float:
        """Total spend so far in USD."""
        with self._lock:
            return sum(r.cost_usd for r in self._records)

    @property
    def remaining_budget(self) -> float:
        """Remaining budget in USD."""
        return max(0.0, self._budget - self.total_cost)

    @property
    def budget_exhausted(self) -> bool:
        """True when the total cost has exceeded the budget."""
        return self.total_cost >= self._budget

    def estimate_cost_per_video(
        self,
        style: str = "cinematic",
        duration: int = 30,
    ) -> Dict[str, float]:
        """Estimate the cost components for a single video.

        Args:
            style: Video style (``"motion"``, ``"cinematic"``, ``"avatar"``).
            duration: Video duration in seconds.

        Returns:
            Dict of ``{"component": cost_usd, …, "total": total_usd}``.
        """
        scene_count = max(3, duration // 10)
        breakdown: Dict[str, float] = {}

        # Gemini script (~600 tokens in / ~400 out)
        breakdown["script_gemini"] = (
            600 / 1000 * self._costs["gemini"]["input_per_1k_tokens"]
            + 400 / 1000 * self._costs["gemini"]["output_per_1k_tokens"]
        )

        if style in ("motion", "cinematic"):
            # One image per scene.
            breakdown["images_sd"] = (
                scene_count * self._costs["stable_diffusion"]["per_image"]
            )
            # Runway clip per scene.
            breakdown["video_runway"] = (
                scene_count * 5 * self._costs["runway"]["per_second"]
            )
        elif style == "avatar":
            # HeyGen avatar video.
            breakdown["video_heygen"] = (
                (duration / 60.0) * self._costs["heygen"]["per_minute"]
            )
            # Wav2Lip.
            breakdown["wav2lip"] = self._costs["wav2lip"]["per_run"]

        # TTS (approx 150 words × 5 chars, Google TTS).
        approx_chars = duration * 15
        breakdown["tts_google"] = (
            approx_chars / 1_000_000 * self._costs["google_tts"]["per_1m_chars"]
        )

        breakdown["total"] = sum(breakdown.values())
        return {k: round(v, 4) for k, v in breakdown.items()}

    def get_summary(self) -> CostSummary:
        """Return a :class:`CostSummary` snapshot.

        Returns:
            Aggregated cost summary.
        """
        with self._lock:
            records = list(self._records)
            videos = self._videos_generated

        total = sum(r.cost_usd for r in records)
        by_provider: Dict[str, float] = {}
        by_operation: Dict[str, float] = {}
        for r in records:
            by_provider[r.provider] = by_provider.get(r.provider, 0.0) + r.cost_usd
            by_operation[r.operation] = (
                by_operation.get(r.operation, 0.0) + r.cost_usd
            )

        return CostSummary(
            total_usd=round(total, 4),
            budget_usd=self._budget,
            remaining_usd=round(max(0.0, self._budget - total), 4),
            usage_pct=round(total / self._budget * 100, 1),
            by_provider={k: round(v, 4) for k, v in by_provider.items()},
            by_operation={k: round(v, 4) for k, v in by_operation.items()},
            record_count=len(records),
            videos_generated=videos,
            cost_per_video=round(total / videos, 4) if videos else 0.0,
        )

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_report(self, filename: Optional[str] = None) -> Path:
        """Serialise usage records to a JSON file.

        Args:
            filename: Output file name (auto-generated if omitted).

        Returns:
            Path to the written file.
        """
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = filename or f"cost_report_{ts}.json"
        path = self._log_dir / fname
        summary = self.get_summary()
        data = {
            "summary": {
                "total_usd": summary.total_usd,
                "budget_usd": summary.budget_usd,
                "remaining_usd": summary.remaining_usd,
                "usage_pct": summary.usage_pct,
                "videos_generated": summary.videos_generated,
                "cost_per_video": summary.cost_per_video,
            },
            "by_provider": summary.by_provider,
            "by_operation": summary.by_operation,
            "records": [r.to_dict() for r in self._records],
        }
        write_json(data, path)
        log.info("Cost report saved: %s", path)
        return path

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _record(
        self,
        provider: str,
        operation: str,
        units: float,
        unit_name: str,
        cost: float,
        metadata: Dict[str, Any],
    ) -> float:
        """Append a usage record and check budget thresholds."""
        rec = APIUsageRecord(
            timestamp=time.time(),
            provider=provider,
            operation=operation,
            units=units,
            unit_name=unit_name,
            cost_usd=cost,
            metadata=metadata,
        )
        with self._lock:
            self._records.append(rec)
            total = sum(r.cost_usd for r in self._records)

        usage_pct = total / self._budget
        if not self._alert_sent and usage_pct >= self._alert_threshold:
            self._alert_sent = True
            log.warning(
                "⚠️  Budget alert: %.0f%% used ($%.2f / $%.2f).",
                usage_pct * 100,
                total,
                self._budget,
            )

        if total >= self._budget:
            log.error(
                "🛑 Budget exhausted! Total: $%.2f ≥ limit $%.2f.",
                total,
                self._budget,
            )

        return cost
