"""
utils/logger.py - Structured logging with file rotation and console output.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

try:
    import colorlog  # type: ignore
    _HAS_COLORLOG = True
except ImportError:
    _HAS_COLORLOG = False

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 10 * 1024 * 1024   # 10 MB
_BACKUP_COUNT = 5
_DEFAULT_LOG_DIR = Path("output/logs")


def _get_console_handler(level: int) -> logging.Handler:
    if _HAS_COLORLOG:
        fmt = colorlog.ColoredFormatter(
            "%(log_color)s" + _LOG_FORMAT,
            datefmt=_DATE_FORMAT,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    else:
        fmt = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(fmt)
    return handler


def _get_file_handler(log_dir: Path, name: str, level: int) -> logging.Handler:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    handler = RotatingFileHandler(
        log_path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    return handler


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """Return a named logger with console and/or rotating-file handlers.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).
        level: Minimum log level (default: ``logging.INFO``).
        log_dir: Directory for log files.  Defaults to ``output/logs``.
        console: Whether to attach a console handler.
        file: Whether to attach a rotating-file handler.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured – return as-is to avoid duplicate handlers.
        return logger

    logger.setLevel(level)
    log_dir = log_dir or _DEFAULT_LOG_DIR

    if console:
        logger.addHandler(_get_console_handler(level))
    if file:
        logger.addHandler(_get_file_handler(log_dir, name, level))

    logger.propagate = False
    return logger
