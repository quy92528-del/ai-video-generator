"""Batch processor for AI video generation.

Reads a list of video topics (from a CSV or JSON file), uses
:class:`~src.gemini_client.GeminiClient` to expand each topic into a
detailed video prompt, then passes the prompt to
:class:`~src.video_generator.VideoGenerator` to create the video.
"""

import csv
import json
import logging
import os
from pathlib import Path
from typing import Generator

from tqdm import tqdm

from src.gemini_client import GeminiClient
from src.video_generator import VideoGenerator, VideoGenerationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> list[dict]:
    """Load rows from a CSV file.

    Expected columns: ``topic`` (required), ``style`` (optional).
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def _load_json(path: str) -> list[dict]:
    """Load rows from a JSON file (list of objects)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
    return data


def load_prompts(input_file: str) -> list[dict]:
    """Load video topics from *input_file* (CSV or JSON)."""
    ext = Path(input_file).suffix.lower()
    if ext == ".csv":
        return _load_csv(input_file)
    if ext == ".json":
        return _load_json(input_file)
    raise ValueError(f"Unsupported input file format: {ext!r} (use .csv or .json)")


# ---------------------------------------------------------------------------
# Batch processor
# ---------------------------------------------------------------------------

class BatchProcessor:
    """Orchestrate batch video generation for a list of topics.

    Parameters
    ----------
    gemini_client:
        Initialised :class:`GeminiClient` used to expand topics into prompts.
    video_generator:
        Initialised :class:`VideoGenerator` used to produce the video files.
    batch_size:
        Maximum number of videos to generate in a single run.  Use ``0`` or
        a negative value to process all rows.
    """

    def __init__(
        self,
        gemini_client: GeminiClient,
        video_generator: VideoGenerator,
        batch_size: int = 10,
    ):
        self.gemini_client = gemini_client
        self.video_generator = video_generator
        self.batch_size = batch_size

    # ------------------------------------------------------------------

    def run(self, input_file: str) -> dict:
        """Process *input_file* and generate up to *batch_size* videos.

        Returns a summary dict with keys ``total``, ``succeeded``, ``failed``.
        """
        rows = load_prompts(input_file)

        # Honour batch_size limit
        if self.batch_size > 0:
            rows = rows[: self.batch_size]

        summary = {"total": len(rows), "succeeded": 0, "failed": 0, "errors": []}

        for idx, row in enumerate(
            tqdm(rows, desc="Generating videos", unit="video"), start=1
        ):
            topic = row.get("topic", "").strip()
            style = row.get("style", "cinematic").strip() or "cinematic"
            filename = row.get("filename", f"video_{idx:04d}.mp4").strip()

            if not topic:
                logger.warning("Row %d has no 'topic' — skipping.", idx)
                summary["failed"] += 1
                summary["errors"].append({"row": idx, "reason": "missing topic"})
                continue

            try:
                logger.info("[%d/%d] Generating prompt for: %s", idx, len(rows), topic)
                video_prompt = self.gemini_client.generate_video_prompt(topic, style)
                logger.debug("Prompt: %s", video_prompt)

                output_path = self.video_generator.generate(video_prompt, filename)
                logger.info("Saved → %s", output_path)
                summary["succeeded"] += 1

            except VideoGenerationError as exc:
                logger.error("Video generation failed for row %d: %s", idx, exc)
                summary["failed"] += 1
                summary["errors"].append({"row": idx, "topic": topic, "reason": str(exc)})

            except (KeyboardInterrupt, SystemExit):
                raise

            except Exception as exc:  # noqa: BLE001
                logger.error("Unexpected error for row %d: %s", idx, exc)
                summary["failed"] += 1
                summary["errors"].append({"row": idx, "topic": topic, "reason": str(exc)})

        return summary

    # ------------------------------------------------------------------

    def iter_rows(self, input_file: str) -> Generator[dict, None, None]:
        """Yield individual row dicts from *input_file* (no batch limit)."""
        yield from load_prompts(input_file)
