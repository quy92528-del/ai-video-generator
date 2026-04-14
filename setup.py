"""
setup.py - Installation script for AI Video Generator Suite.

Usage:
    python setup.py install         # Install package
    python setup.py develop         # Install in dev/editable mode
    python setup.py setup_dirs      # Create required directory structure
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.command.develop import develop

# ---------------------------------------------------------------------------
# Package Metadata
# ---------------------------------------------------------------------------

NAME = "ai-video-generator"
VERSION = "1.0.0"
DESCRIPTION = "AI automation tool to generate 500+ videos using Gemini, Runway, and more."
AUTHOR = "AI Video Generator Team"
PYTHON_REQUIRES = ">=3.10"

INSTALL_REQUIRES = [
    "streamlit>=1.28",
    "python-dotenv>=1.0",
    "requests>=2.31",
    "pydantic>=2.4",
    "pydantic-settings>=2.1",
    "google-generativeai>=0.3",
    "google-cloud-texttospeech>=2.14",
    "moviepy>=1.0.3",
    "opencv-python-headless>=4.8",
    "Pillow>=10.1",
    "numpy>=1.24",
    "scipy>=1.11",
    "librosa>=0.10",
    "soundfile>=0.12",
    "pydub>=0.25",
    "mediapipe>=0.10",
    "onnxruntime>=1.16",
    "aiohttp>=3.9",
    "httpx>=0.25",
    "tenacity>=8.2",
    "redis>=5.0",
    "celery>=5.3",
    "replicate>=0.22",
    "tqdm>=4.66",
    "PyYAML>=6.0",
    "psutil>=5.9",
    "rich>=13.7",
    "colorlog>=6.8",
    "cachetools>=5.3",
]

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

REQUIRED_DIRS = [
    "output/videos",
    "output/thumbnails",
    "output/metadata",
    "output/logs",
    "assets/music",
    "assets/sounds",
    "assets/fonts",
    "assets/templates",
    ".cache",
]


def _create_dirs() -> None:
    for d in REQUIRED_DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)
        gitkeep = Path(d) / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
    print("✅ Directory structure created.")


def _copy_env_example() -> None:
    src = Path(".env.example")
    dst = Path(".env")
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        print("✅ .env created from .env.example – please fill in your API keys.")
    elif not dst.exists():
        print("⚠️  .env.example not found – please create .env manually.")


# ---------------------------------------------------------------------------
# Custom install commands
# ---------------------------------------------------------------------------

class PostInstall(install):  # type: ignore[misc]
    """Run post-install tasks after package installation."""

    def run(self) -> None:
        install.run(self)
        _create_dirs()
        _copy_env_example()


class PostDevelop(develop):  # type: ignore[misc]
    """Run post-develop tasks after editable install."""

    def run(self) -> None:
        develop.run(self)
        _create_dirs()
        _copy_env_example()


# ---------------------------------------------------------------------------
# CLI entry point (python setup.py setup_dirs)
# ---------------------------------------------------------------------------

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "setup_dirs":
    _create_dirs()
    _copy_env_example()
    print("\n🎬 AI Video Generator Suite is ready.")
    print("   1. Edit .env and add your API keys.")
    print("   2. Run:  pip install -r requirements.txt")
    print("   3. Run:  streamlit run main.py")
    sys.exit(0)

# ---------------------------------------------------------------------------
# setup() call
# ---------------------------------------------------------------------------

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    python_requires=PYTHON_REQUIRES,
    packages=find_packages(exclude=["tests*", "output*", "assets*"]),
    install_requires=INSTALL_REQUIRES,
    cmdclass={
        "install": PostInstall,
        "develop": PostDevelop,
    },
    entry_points={
        "console_scripts": [
            "ai-video-generator=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
