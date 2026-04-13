#!/usr/bin/env python3
"""AI Video Generator — main entry point.

Usage
-----
    python main.py [OPTIONS]

Options
-------
  --input FILE      Path to input CSV or JSON file with video topics.
                    Defaults to INPUT_FILE env var or 'prompts.csv'.
  --output DIR      Directory where generated videos will be saved.
                    Defaults to OUTPUT_DIR env var or 'output'.
  --batch-size N    Maximum number of videos to generate per run.
                    Defaults to BATCH_SIZE env var or 10.
  --model NAME      Gemini model to use for prompt generation.
                    Defaults to GEMINI_MODEL env var or 'gemini-1.5-pro'.
  --log-level LVL   Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.
  --dry-run         Print the expanded prompts without generating videos.

Environment variables
---------------------
  GEMINI_API_KEY    Required.  Your Google Gemini API key.
  VEO_API_KEY       Optional.  Video generation API key (Veo / compatible).
  INPUT_FILE        Default input file path.
  OUTPUT_DIR        Default output directory.
  BATCH_SIZE        Default batch size.
  GEMINI_MODEL      Default Gemini model name.

Example
-------
    # generate 20 videos described in prompts.csv
    python main.py --input prompts.csv --batch-size 20

    # dry run — preview prompts without calling the video API
    python main.py --input prompts.json --dry-run
"""

import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv

# Load variables from .env (if present) before anything else
load_dotenv()

from src.batch_processor import BatchProcessor, load_prompts  # noqa: E402
from src.gemini_client import GeminiClient  # noqa: E402
from src.video_generator import VideoGenerator  # noqa: E402


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Batch-generate AI videos using the Gemini API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default=os.environ.get("INPUT_FILE", "prompts.csv"),
        metavar="FILE",
        help="Input CSV or JSON file with video topics (default: prompts.csv).",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("OUTPUT_DIR", "output"),
        metavar="DIR",
        help="Output directory for generated videos (default: output).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("BATCH_SIZE", 10)),
        metavar="N",
        help="Maximum videos to generate per run (default: 10).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_MODEL", "gemini-1.5-pro"),
        metavar="NAME",
        help="Gemini model name (default: gemini-1.5-pro).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        metavar="LEVEL",
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print expanded prompts without calling the video generation API.",
    )
    return parser


# ---------------------------------------------------------------------------
# Dry-run helper
# ---------------------------------------------------------------------------

def _dry_run(args: argparse.Namespace, gemini: GeminiClient) -> None:
    """Print the Gemini-expanded prompts for every row in the input file."""
    rows = load_prompts(args.input)
    if args.batch_size > 0:
        rows = rows[: args.batch_size]

    print(f"\nDry run — {len(rows)} topic(s) from '{args.input}'\n")
    for idx, row in enumerate(rows, start=1):
        topic = row.get("topic", "").strip()
        style = row.get("style", "cinematic").strip() or "cinematic"
        if not topic:
            print(f"  [{idx}] (skipped — missing topic)")
            continue
        prompt = gemini.generate_video_prompt(topic, style)
        print(f"  [{idx}] Topic : {topic}")
        print(f"       Style : {style}")
        print(f"       Prompt: {prompt}")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Validate that the required API key is present
    if not os.environ.get("GEMINI_API_KEY"):
        logger.error(
            "GEMINI_API_KEY is not set. "
            "Copy .env.example to .env and add your key, or export it directly."
        )
        return 1

    gemini = GeminiClient(model=args.model)

    if args.dry_run:
        _dry_run(args, gemini)
        return 0

    video_gen = VideoGenerator(output_dir=args.output)
    processor = BatchProcessor(
        gemini_client=gemini,
        video_generator=video_gen,
        batch_size=args.batch_size,
    )

    logger.info(
        "Starting batch: input=%s  output=%s  batch_size=%d  model=%s",
        args.input,
        args.output,
        args.batch_size,
        args.model,
    )

    summary = processor.run(args.input)

    # Print summary
    print("\n--- Batch complete ---")
    print(f"  Total   : {summary['total']}")
    print(f"  Success : {summary['succeeded']}")
    print(f"  Failed  : {summary['failed']}")

    if summary["errors"]:
        print("\nErrors:")
        print(json.dumps(summary["errors"], indent=2))

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
