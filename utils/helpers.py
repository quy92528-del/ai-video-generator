"""
utils/helpers.py - Common utility functions: file operations, data transforms.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# String Helpers
# ---------------------------------------------------------------------------

def slugify(text: str, max_length: int = 60) -> str:
    """Convert *text* to a filesystem-safe slug.

    Args:
        text: Input string.
        max_length: Maximum slug length.

    Returns:
        Lowercase slug with non-alphanumeric chars replaced by hyphens.
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_length]


def truncate(text: str, max_chars: int = 100, suffix: str = "…") -> str:
    """Truncate *text* to *max_chars* adding *suffix* if truncated."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix


# ---------------------------------------------------------------------------
# File & Directory Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and parents) if it does not exist.

    Args:
        path: Directory path.

    Returns:
        Resolved :class:`Path` object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_delete(path: str | Path) -> bool:
    """Delete a file or directory tree without raising if missing.

    Args:
        path: Target path.

    Returns:
        ``True`` if the path was deleted, ``False`` if it didn't exist.
    """
    p = Path(path)
    if p.is_file():
        p.unlink()
        return True
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
        return True
    return False


def write_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """Serialise *data* to a JSON file.

    Args:
        data: JSON-serialisable object.
        path: Output file path.
        indent: JSON indentation level.

    Returns:
        Resolved :class:`Path` of the written file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")
    return p


def read_json(path: str | Path) -> Any:
    """Read and parse a JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed Python object.

    Raises:
        FileNotFoundError: If *path* does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unique_filename(directory: str | Path, stem: str, suffix: str) -> Path:
    """Return a unique file path inside *directory*.

    Appends a short UUID if *stem + suffix* already exists.

    Args:
        directory: Target directory.
        stem: Base file name without extension.
        suffix: File extension including the dot (e.g. ``".mp4"``).

    Returns:
        A :class:`Path` that does not exist yet.
    """
    base = Path(directory) / f"{stem}{suffix}"
    if not base.exists():
        return base
    short_id = uuid.uuid4().hex[:8]
    return Path(directory) / f"{stem}_{short_id}{suffix}"


# ---------------------------------------------------------------------------
# Iterable Helpers
# ---------------------------------------------------------------------------

def chunk_list(items: List[T], size: int) -> Generator[List[T], None, None]:
    """Yield successive *size*-length chunks from *items*.

    Args:
        items: Source list.
        size: Chunk size (must be ≥ 1).

    Yields:
        Sub-lists of length ≤ *size*.
    """
    if size < 1:
        raise ValueError("Chunk size must be at least 1.")
    for i in range(0, len(items), size):
        yield items[i : i + size]


def flatten(nested: Iterable[Iterable[T]]) -> List[T]:
    """Flatten one level of nesting.

    Args:
        nested: An iterable of iterables.

    Returns:
        A flat list.
    """
    return [item for sublist in nested for item in sublist]


# ---------------------------------------------------------------------------
# Hashing / ID Helpers
# ---------------------------------------------------------------------------

def content_hash(text: str) -> str:
    """Return a short SHA-256 hex digest of *text* (first 16 chars).

    Useful as a cache key for prompts / scripts.

    Args:
        text: Input string.

    Returns:
        16-character hex string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def generate_job_id(prefix: str = "job") -> str:
    """Generate a unique job identifier.

    Args:
        prefix: Short prefix string.

    Returns:
        String like ``"job_20240101_120000_a1b2c3d4"``.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"{prefix}_{ts}_{short}"


# ---------------------------------------------------------------------------
# Time Helpers
# ---------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        String like ``"2m 30s"`` or ``"45s"``.
    """
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins:02d}m {secs:02d}s"


class Timer:
    """Simple context-manager wall-clock timer.

    Usage::

        with Timer() as t:
            do_work()
        print(f"Elapsed: {t.elapsed:.2f}s")
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed = time.perf_counter() - self._start


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def create_output_dirs(base_dir: str | Path = "output") -> Dict[str, Path]:
    """Create the standard output directory structure.

    Args:
        base_dir: Root output directory (default: ``"output"``).

    Returns:
        Dictionary mapping directory names to resolved :class:`Path` objects.
    """
    base = Path(base_dir)
    dirs = {
        "root": base,
        "videos": base / "videos",
        "thumbnails": base / "thumbnails",
        "metadata": base / "metadata",
        "logs": base / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def get_safe_filename(name: str, max_length: int = 100) -> str:
    """Sanitize *name* for use as a filename.

    Strips path separators, replaces whitespace with underscores, and removes
    characters that are unsafe on common operating systems.

    Args:
        name: Raw filename string (without extension).
        max_length: Maximum returned length.

    Returns:
        A sanitized filename string.
    """
    name = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "", name)
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:max_length] or "file"


def list_files(directory: str | Path, pattern: str = "*") -> List[Path]:
    """Return sorted list of files in *directory* matching *pattern*.

    Args:
        directory: Directory to search.
        pattern: Glob pattern (e.g. ``"*.mp4"``).

    Returns:
        Sorted list of matching :class:`Path` objects (files only).
    """
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob(pattern) if p.is_file())


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into a copy of *base*.

    Args:
        base: Base dictionary.
        override: Dictionary with override values.

    Returns:
        Merged dictionary (does not mutate inputs).
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
