"""
cli.py - Command-line batch processing interface for AI Video Generator Suite.

Usage examples::

    python cli.py --topic "Top 5 coffee shops in Hanoi" --count 10
    python cli.py --topic "Nature scenes" --count 100 --batch 10 --style cinematic
    python cli.py --topic "Tech tips" --count 5 --language en --style motion
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from utils.logger import setup_logger
from utils.helpers import create_output_dirs, format_duration, Timer
from utils.validators import (
    validate_batch_size,
    validate_topic,
    validate_video_count,
)

log = setup_logger()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="AI Video Generator – command-line batch processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python cli.py --topic "Coffee shops in Hanoi" --count 10
  python cli.py --topic "Nature" --count 50 --batch 5 --style cinematic
  python cli.py --topic "Tech tips" --count 5 --language en
        """,
    )

    parser.add_argument(
        "--topic",
        required=True,
        help="Video topic / subject (required)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Total number of videos to generate (1-500, default: 10)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=10,
        help="Number of videos to process per batch (1-100, default: 10)",
    )
    parser.add_argument(
        "--style",
        choices=["motion", "cinematic", "avatar"],
        default="cinematic",
        help="Video style (default: cinematic)",
    )
    parser.add_argument(
        "--language",
        default="vi",
        help="Output language code, e.g. en, vi, zh (default: vi)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Video duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Root output directory (default: output)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show plan without generating videos",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose / debug logging",
    )

    return parser


# ---------------------------------------------------------------------------
# Core batch logic
# ---------------------------------------------------------------------------

def _run_batch(
    topic: str,
    count: int,
    batch_size: int,
    style: str,
    language: str,
    duration: int,
    output_dir: str,
) -> int:
    """Run the batch generation pipeline.

    Args:
        topic: Video topic.
        count: Total videos to generate.
        batch_size: Videos per batch.
        style: Video style string.
        language: Language code.
        duration: Video duration in seconds.
        output_dir: Root output directory.

    Returns:
        Number of successfully generated videos.
    """
    from config import get_settings
    from brain_module import BrainModule
    from utils.helpers import chunk_list

    settings = get_settings()

    # Ensure output directories exist.
    dirs = create_output_dirs(output_dir)
    log.info(f"Output directory: {dirs['root'].resolve()}")

    # Initialize BrainModule for script generation.
    brain = BrainModule(api_key=settings.google_gemini_api_key)

    log.info(
        f"Starting batch generation: topic='{topic}' count={count} "
        f"batch={batch_size} style={style} language={language} duration={duration}s"
    )

    # Generate scripts for all videos upfront.
    log.info("Generating scripts …")
    try:
        scripts = brain.generate_batch(
            topic=topic,
            count=count,
            styles=[style],
            language=language,
            duration=duration,
        )
    except Exception as exc:  # pylint: disable=broad-except
        log.error(f"Script generation failed: {exc}")
        return 0

    if not scripts:
        log.warning("No scripts were generated – aborting")
        return 0

    log.info(f"Generated {len(scripts)} scripts")

    # Process in batches.
    success = 0
    for batch_num, batch in enumerate(chunk_list(scripts, batch_size), start=1):
        log.info(f"Processing batch {batch_num} ({len(batch)} videos) …")
        for script in batch:
            try:
                log.info(f"  → {script.title}")
                # TODO: Wire up VisualEngine, AudioModule, PostProduction when
                # API keys are configured.  Scripts are written to metadata/.
                from utils.helpers import write_json
                meta_path = dirs["metadata"] / f"{script.cache_key or 'video'}.json"
                write_json(script.to_dict(), meta_path)
                success += 1
            except Exception as exc:  # pylint: disable=broad-except
                log.error(f"Failed to process '{script.title}': {exc}")

    return success


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    import logging

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        log.setLevel(logging.DEBUG)

    # --- Validate inputs ----------------------------------------------------
    try:
        topic = validate_topic(args.topic)
        count = validate_video_count(args.count)
        batch_size = validate_batch_size(args.batch)
    except ValueError as exc:
        log.error(f"Invalid argument: {exc}")
        return 1

    log.info("=" * 60)
    log.info("AI Video Generator – CLI Batch Processor")
    log.info("=" * 60)
    log.info(f"Topic    : {topic}")
    log.info(f"Count    : {count}")
    log.info(f"Batch    : {batch_size}")
    log.info(f"Style    : {args.style}")
    log.info(f"Language : {args.language}")
    log.info(f"Duration : {args.duration}s")
    log.info(f"Output   : {args.output_dir}")

    if args.dry_run:
        log.info("[DRY RUN] Validation passed – no videos will be generated")
        return 0

    # --- Run -----------------------------------------------------------------
    with Timer() as timer:
        generated = _run_batch(
            topic=topic,
            count=count,
            batch_size=batch_size,
            style=args.style,
            language=args.language,
            duration=args.duration,
            output_dir=args.output_dir,
        )

    log.info("=" * 60)
    log.info(
        f"Done – {generated}/{count} videos processed in {format_duration(timer.elapsed)}"
    )

    return 0 if generated > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
