"""
cli.py - Command-line entry point for AI Video Generator Suite.

Usage:
    python cli.py --topic "Vietnam travel" --count 5
    python cli.py --topic "My Topic" --count 10 --batch 5 --style cinematic
    python cli.py --topic "Hanoi food" --count 1 --language en --duration 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

from config import (
    SUPPORTED_ASPECT_RATIOS,
    SUPPORTED_LANGUAGES,
    SUPPORTED_VIDEO_DURATIONS,
    SUPPORTED_VIDEO_STYLES,
    get_settings,
)
from utils.logger import get_logger
from utils.helpers import ensure_dir, write_json, format_duration, chunk_list
from utils.validators import (
    validate_config,
    validate_topic,
    validate_video_count,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="AI Video Generator Suite – batch video generation CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--topic",
        required=True,
        help="Topic or theme for the generated videos (e.g. 'Top 5 coffee shops in Hanoi').",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        metavar="N",
        help="Number of videos to generate (1–500).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=10,
        metavar="SIZE",
        help="Number of videos to process per batch.",
    )
    parser.add_argument(
        "--style",
        choices=list(SUPPORTED_VIDEO_STYLES),
        default="cinematic",
        help="Visual style for all generated videos.",
    )
    parser.add_argument(
        "--language",
        choices=list(SUPPORTED_LANGUAGES.keys()),
        default="vi",
        help="Narration and script language code.",
    )
    parser.add_argument(
        "--aspect-ratio",
        dest="aspect_ratio",
        choices=list(SUPPORTED_ASPECT_RATIOS.keys()),
        default="9:16",
        help="Output video aspect ratio.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        choices=list(SUPPORTED_VIDEO_DURATIONS),
        default=30,
        metavar="SECONDS",
        help=f"Duration of each video in seconds. Allowed: {SUPPORTED_VIDEO_DURATIONS}.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Override output directory (defaults to settings.output_dir).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and settings without generating any videos.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging.",
    )

    return parser


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Validate parsed CLI arguments and return a normalised params dict.

    Args:
        args: Parsed :class:`argparse.Namespace`.

    Returns:
        Normalised parameters dictionary.

    Raises:
        SystemExit: When validation fails (prints an error and exits with code 2).
    """
    try:
        topic = validate_topic(args.topic)
        count = validate_video_count(args.count)
    except ValueError as exc:
        log.error("Validation error: %s", exc)
        sys.exit(2)

    if args.batch < 1:
        log.error("Batch size must be at least 1.")
        sys.exit(2)

    return {
        "topic": topic,
        "count": count,
        "batch": args.batch,
        "style": args.style,
        "language": args.language,
        "aspect_ratio": args.aspect_ratio,
        "duration": args.duration,
        "output_dir": args.output_dir,
        "dry_run": args.dry_run,
        "verbose": args.verbose,
    }


# ---------------------------------------------------------------------------
# Generation Pipeline
# ---------------------------------------------------------------------------

def _run_batch(
    params: Dict[str, Any],
    batch_indices: List[int],
    settings: Any,
    output_dir: str,
    results: List[Dict[str, Any]],
) -> None:
    """Run generation for one batch of videos.

    Args:
        params: Normalised parameters from :func:`_validate_args`.
        batch_indices: 1-based indices of the videos in this batch.
        settings: Loaded :class:`config.Settings` instance.
        output_dir: Resolved output directory path.
        results: List to append result dicts to (mutated in place).
    """
    from brain_module import BrainModule
    from visual_engine import VisualEngine
    from audio_module import AudioModule
    from post_production import PostProduction
    from cost_tracker import CostTracker

    brain = BrainModule(
        api_key=settings.google_gemini_api_key,
        model=settings.gemini_model,
    )
    visual = VisualEngine(
        replicate_token=settings.replicate_api_token,
        runway_api_key=settings.runway_api_key,
        heygen_api_key=settings.heygen_api_key,
        output_dir=output_dir,
    )
    audio = AudioModule(
        google_credentials_json=settings.google_cloud_credentials_json,
        elevenlabs_api_key=settings.elevenlabs_api_key,
        replicate_token=settings.replicate_api_token,
        output_dir=output_dir,
    )
    post = PostProduction(output_dir=output_dir, fps=settings.video_fps)
    tracker = CostTracker(budget_usd=settings.budget_limit_usd)

    total = params["count"]

    for idx in batch_indices:
        if tracker.budget_exhausted:
            log.warning("Budget exhausted – stopping generation.")
            break

        log.info("[%d/%d] Generating %s video …", idx, total, params["style"])

        try:
            # 1. Generate script.
            script = brain.generate_script(
                topic=params["topic"],
                style=params["style"],
                language=params["language"],
                duration=params["duration"],
                variation_index=idx - 1,
            )
            tracker.record_gemini(input_tokens=600, output_tokens=400)

            # 2. Generate scenes.
            clip_paths: List[str] = []
            for scene in script.scenes:
                try:
                    if params["style"] == "avatar":
                        clip = visual.create_avatar_video(
                            script_text=scene.narration,
                            output_dir=f"{output_dir}/clips",
                        )
                        tracker.record_video_generation(
                            clip.duration_seconds, provider="heygen"
                        )
                    elif params["style"] == "cinematic":
                        clip = visual.generate_cinematic_video(
                            prompt=scene.description,
                            duration_seconds=scene.duration_seconds,
                            output_dir=f"{output_dir}/clips",
                        )
                        tracker.record_video_generation(
                            clip.duration_seconds, provider="runway"
                        )
                    else:  # motion
                        img = visual.generate_image(
                            prompt=scene.description,
                            output_dir=f"{output_dir}/images",
                        )
                        tracker.record_image_generation(1)
                        clip = visual.generate_video_from_image(
                            image_path=img.path,
                            prompt=scene.description,
                            duration_seconds=scene.duration_seconds,
                            output_dir=f"{output_dir}/clips",
                        )
                        tracker.record_video_generation(
                            clip.duration_seconds, provider="runway"
                        )
                    clip_paths.append(clip.path)
                except Exception as exc:  # pylint: disable=broad-except
                    log.warning("  ⚠️ Scene %d failed: %s", scene.index, exc)

            # 3. Generate audio.
            all_narration = " ".join(s.narration for s in script.scenes)
            voice_track = audio.synthesize_speech(
                text=all_narration,
                language=params["language"],
            )
            tracker.record_tts(len(all_narration), provider="google_tts")
            mixed = audio.mix_audio(voice_track=voice_track)

            # 4. Assemble video.
            final = post.assemble(
                clip_paths=clip_paths,
                audio_path=mixed.path,
                output_filename=f"video_{idx:04d}.mp4",
                aspect_ratio=params["aspect_ratio"],
            )

            # 5. Write metadata.
            meta: Dict[str, Any] = {
                "index": idx,
                "title": script.title,
                "topic": params["topic"],
                "style": params["style"],
                "language": params["language"],
                "duration": params["duration"],
                "aspect_ratio": params["aspect_ratio"],
                "cost_usd": tracker.total_cost,
                "output_path": str(final.path),
            }
            meta_path = write_json(
                meta, f"{output_dir}/metadata/video_{idx:04d}.json"
            )
            log.info("  ✓ Video %d complete → %s", idx, final.path)
            results.append(meta)

        except Exception as exc:  # pylint: disable=broad-except
            log.error("  ✗ Video %d failed: %s", idx, exc)
            results.append({"index": idx, "error": str(exc)})


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run batch video generation."""
    parser = _build_parser()
    args = parser.parse_args()

    # Configure log level.
    if args.verbose:
        import logging
        log.setLevel(logging.DEBUG)

    log.info("=" * 60)
    log.info("AI Video Generator Suite – CLI")
    log.info("=" * 60)

    # Load and validate settings.
    try:
        settings = get_settings()
    except Exception as exc:  # pylint: disable=broad-except
        log.error("Failed to load settings: %s", exc)
        sys.exit(1)

    config_warnings = validate_config(settings)
    if config_warnings:
        for w in config_warnings:
            log.warning(w)

    # Validate CLI arguments.
    params = _validate_args(args)

    # Resolve output directory.
    output_dir = params["output_dir"] or settings.output_dir
    ensure_dir(output_dir)
    ensure_dir(f"{output_dir}/videos")
    ensure_dir(f"{output_dir}/metadata")

    log.info("Topic        : %s", params["topic"])
    log.info("Count        : %d", params["count"])
    log.info("Batch size   : %d", params["batch"])
    log.info("Style        : %s", params["style"])
    log.info("Language     : %s", params["language"])
    log.info("Aspect ratio : %s", params["aspect_ratio"])
    log.info("Duration     : %ds", params["duration"])
    log.info("Output dir   : %s", output_dir)

    if params["dry_run"]:
        log.info("Dry-run mode – no videos will be generated.")
        sys.exit(0)

    # Split into batches and process.
    all_indices = list(range(1, params["count"] + 1))
    batches = list(chunk_list(all_indices, params["batch"]))
    results: List[Dict[str, Any]] = []

    for batch_num, batch_indices in enumerate(batches, start=1):
        log.info(
            "--- Batch %d/%d (videos %d–%d) ---",
            batch_num,
            len(batches),
            batch_indices[0],
            batch_indices[-1],
        )
        _run_batch(params, batch_indices, settings, output_dir, results)

    # Summary.
    succeeded = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    log.info("=" * 60)
    log.info(
        "Done: %d succeeded, %d failed out of %d requested.",
        len(succeeded),
        len(failed),
        params["count"],
    )

    # Write run summary.
    summary = {
        "topic": params["topic"],
        "requested": params["count"],
        "succeeded": len(succeeded),
        "failed": len(failed),
        "videos": results,
    }
    summary_path = write_json(summary, f"{output_dir}/metadata/run_summary.json")
    log.info("Run summary written to %s", summary_path)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
