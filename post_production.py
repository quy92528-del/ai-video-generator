"""
post_production.py - Video assembly, transitions, captions, thumbnails, color
grading, and smart re-framing using MoviePy.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger
from utils.helpers import ensure_dir, unique_filename

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class AssembledVideo:
    """Final assembled video ready for distribution."""
    path: Path
    duration_seconds: float
    width: int
    height: int
    thumbnail_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PostProduction
# ---------------------------------------------------------------------------

class PostProduction:
    """Assemble, grade, caption, and package AI-generated video clips.

    Args:
        output_dir: Root output directory.
        fps: Output frames per second.
    """

    def __init__(self, output_dir: str = "output", fps: int = 30) -> None:
        self._output_dir = Path(output_dir)
        self._fps = fps

    # ------------------------------------------------------------------ #
    # Main Assembly Pipeline                                               #
    # ------------------------------------------------------------------ #

    def assemble_video(
        self,
        clip_paths: List[Path],
        audio_path: Optional[Path] = None,
        aspect_ratio: str = "9:16",
        title: str = "",
        captions: Optional[List[Dict[str, Any]]] = None,
        transition: str = "fade",
        color_grade: str = "cinematic",
        output_dir: Optional[str] = None,
    ) -> AssembledVideo:
        """Assemble clips into a finished video.

        Args:
            clip_paths: Ordered list of video/image clip file paths.
            audio_path: Optional mixed audio track.
            aspect_ratio: ``"9:16"``, ``"16:9"``, or ``"1:1"``.
            title: Video title overlay (shown for 3 seconds at start).
            captions: List of ``{text, start, end}`` caption dicts.
            transition: Transition type (``"fade"``, ``"cut"``, ``"wipe"``).
            color_grade: Colour grading preset.
            output_dir: Override output directory.

        Returns:
            :class:`AssembledVideo`.
        """
        save_dir = ensure_dir(output_dir or self._output_dir / "videos")
        out_path = unique_filename(save_dir, "video", ".mp4")
        width, height = self._aspect_to_dims(aspect_ratio)

        try:
            return self._assemble_with_moviepy(
                clip_paths,
                audio_path,
                width,
                height,
                title,
                captions or [],
                transition,
                color_grade,
                out_path,
            )
        except ImportError:
            log.warning("MoviePy not installed – creating mock video file.")
            return self._mock_assemble(out_path, width, height)
        except Exception as exc:  # pylint: disable=broad-except
            log.error("Video assembly failed: %s", exc)
            return self._mock_assemble(out_path, width, height)

    def generate_thumbnail(
        self,
        video_path: Path,
        title: str = "",
        at_second: float = 1.0,
        output_dir: Optional[str] = None,
    ) -> Optional[Path]:
        """Extract and annotate a thumbnail from *video_path*.

        Args:
            video_path: Source video file.
            title: Text overlay for the thumbnail.
            at_second: Timestamp to extract the frame from.
            output_dir: Override output directory.

        Returns:
            Path to the saved thumbnail PNG, or ``None`` on failure.
        """
        save_dir = ensure_dir(output_dir or self._output_dir / "thumbnails")
        out_path = unique_filename(save_dir, "thumbnail", ".png")
        try:
            from moviepy.editor import VideoFileClip  # type: ignore

            with VideoFileClip(str(video_path)) as clip:
                t = min(at_second, clip.duration - 0.1)
                frame = clip.get_frame(t)
            img = self._annotate_thumbnail(frame, title)
            img.save(str(out_path))
            log.info("Thumbnail saved: %s", out_path)
            return out_path
        except Exception as exc:  # pylint: disable=broad-except
            log.warning("Thumbnail generation failed: %s", exc)
            return None

    def reframe_for_platform(
        self,
        video_path: Path,
        target_ratio: str,
        output_dir: Optional[str] = None,
    ) -> Path:
        """Re-frame a video for a specific aspect ratio using smart crop.

        Args:
            video_path: Source video file.
            target_ratio: Target aspect ratio string.
            output_dir: Override output directory.

        Returns:
            Path to re-framed video.
        """
        save_dir = ensure_dir(output_dir or self._output_dir / "reframed")
        out_path = unique_filename(save_dir, f"reframed_{target_ratio.replace(':', 'x')}", ".mp4")
        tw, th = self._aspect_to_dims(target_ratio)

        try:
            from moviepy.editor import VideoFileClip  # type: ignore

            with VideoFileClip(str(video_path)) as clip:
                # Centre-crop to target ratio.
                src_w, src_h = clip.size
                target_w = min(src_w, int(src_h * tw / th))
                target_h = min(src_h, int(src_w * th / tw))
                x1 = (src_w - target_w) // 2
                y1 = (src_h - target_h) // 2
                cropped = clip.crop(x1=x1, y1=y1, x2=x1 + target_w, y2=y1 + target_h)
                resized = cropped.resize((tw, th))
                resized.write_videofile(
                    str(out_path),
                    fps=self._fps,
                    codec="libx264",
                    audio_codec="aac",
                    logger=None,
                )
            return out_path
        except Exception as exc:  # pylint: disable=broad-except
            log.error("Reframe failed: %s", exc)
            import shutil

            shutil.copy2(video_path, out_path)
            return out_path

    # ------------------------------------------------------------------ #
    # Internal: MoviePy Assembly                                           #
    # ------------------------------------------------------------------ #

    def _assemble_with_moviepy(
        self,
        clip_paths: List[Path],
        audio_path: Optional[Path],
        width: int,
        height: int,
        title: str,
        captions: List[Dict[str, Any]],
        transition: str,
        color_grade: str,
        out_path: Path,
    ) -> AssembledVideo:
        from moviepy.editor import (  # type: ignore
            AudioFileClip,
            ColorClip,
            CompositeVideoClip,
            ImageClip,
            TextClip,
            VideoFileClip,
            concatenate_videoclips,
        )

        clips = []
        for cp in clip_paths:
            if not cp.exists():
                log.warning("Clip not found: %s – skipping.", cp)
                continue
            suffix = cp.suffix.lower()
            if suffix in (".mp4", ".mov", ".avi", ".mkv"):
                c = VideoFileClip(str(cp)).resize((width, height))
            else:
                # Treat as image – 5 second still.
                c = ImageClip(str(cp), duration=5).resize((width, height))
            c = self._apply_color_grade(c, color_grade)
            clips.append(c)

        if not clips:
            log.error("No valid clips found – aborting assembly.")
            return self._mock_assemble(out_path, width, height)

        final = concatenate_videoclips(clips, method="compose")

        # Add title card.
        if title:
            try:
                txt = (
                    TextClip(
                        title,
                        fontsize=50,
                        color="white",
                        font="Arial",
                        size=(width, None),
                        method="caption",
                    )
                    .set_duration(min(3, final.duration))
                    .set_position("center")
                    .set_start(0)
                    .crossfadein(0.5)
                    .crossfadeout(0.5)
                )
                final = CompositeVideoClip([final, txt])
            except Exception as exc:  # pylint: disable=broad-except
                log.warning("Could not add title overlay: %s", exc)

        # Add captions.
        for cap in captions:
            try:
                ct = (
                    TextClip(
                        cap["text"],
                        fontsize=36,
                        color="white",
                        font="Arial",
                        size=(width - 40, None),
                        method="caption",
                        stroke_color="black",
                        stroke_width=2,
                    )
                    .set_duration(cap["end"] - cap["start"])
                    .set_start(cap["start"])
                    .set_position(("center", height - 200))
                )
                final = CompositeVideoClip([final, ct])
            except Exception as exc:  # pylint: disable=broad-except
                log.warning("Caption error: %s", exc)

        # Attach audio.
        if audio_path and audio_path.exists():
            try:
                audio = AudioFileClip(str(audio_path))
                if audio.duration > final.duration:
                    audio = audio.subclip(0, final.duration)
                final = final.set_audio(audio)
            except Exception as exc:  # pylint: disable=broad-except
                log.warning("Could not attach audio: %s", exc)

        final.write_videofile(
            str(out_path),
            fps=self._fps,
            codec="libx264",
            audio_codec="aac",
            logger=None,
            preset="fast",
        )

        duration = final.duration
        final.close()
        for c in clips:
            c.close()

        thumb_path = self.generate_thumbnail(out_path, title)
        log.info("Video assembled: %s (%.1fs)", out_path, duration)
        return AssembledVideo(
            path=out_path,
            duration_seconds=duration,
            width=width,
            height=height,
            thumbnail_path=thumb_path,
        )

    # ------------------------------------------------------------------ #
    # Colour Grading                                                       #
    # ------------------------------------------------------------------ #

    def _apply_color_grade(self, clip: Any, preset: str) -> Any:
        """Apply a simple colour-grading LUT-equivalent via numpy.

        Args:
            clip: MoviePy clip.
            preset: One of ``"cinematic"``, ``"vivid"``, ``"retro"``,
                    ``"warm"``, ``"cool"``, or ``"none"``.

        Returns:
            Colour-graded clip (or original if numpy unavailable).
        """
        try:
            import numpy as np  # type: ignore

            def _grade(frame: Any, preset: str = preset) -> Any:
                f = frame.astype(np.float32) / 255.0
                if preset == "cinematic":
                    f[..., 0] *= 1.05  # slight red boost
                    f[..., 2] *= 0.95  # slight blue cut
                elif preset == "vivid":
                    f = np.clip(f * 1.2, 0, 1)
                elif preset == "retro":
                    f[..., 1] *= 0.85
                    f = np.clip(f * 1.1, 0, 1)
                elif preset == "warm":
                    f[..., 0] *= 1.1
                    f[..., 2] *= 0.9
                elif preset == "cool":
                    f[..., 0] *= 0.9
                    f[..., 2] *= 1.1
                return np.clip(f * 255, 0, 255).astype(np.uint8)

            return clip.fl_image(_grade)
        except Exception:  # pylint: disable=broad-except
            return clip

    # ------------------------------------------------------------------ #
    # Thumbnail Annotation                                                 #
    # ------------------------------------------------------------------ #

    def _annotate_thumbnail(self, frame: Any, title: str) -> Any:
        """Overlay *title* text on a numpy frame array.

        Returns a Pillow Image.
        """
        from PIL import Image, ImageDraw  # type: ignore

        img = Image.fromarray(frame)
        if title:
            draw = ImageDraw.Draw(img)
            # Simple bottom-centre text overlay.
            w, h = img.size
            draw.rectangle([0, h - 80, w, h], fill=(0, 0, 0, 180))
            draw.text((10, h - 65), title[:50], fill=(255, 255, 255))
        return img

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _aspect_to_dims(ratio: str) -> Tuple[int, int]:
        from config import SUPPORTED_ASPECT_RATIOS

        return SUPPORTED_ASPECT_RATIOS.get(ratio, (1080, 1920))

    @staticmethod
    def _mock_assemble(out_path: Path, width: int, height: int) -> AssembledVideo:
        out_path.write_bytes(b"")
        return AssembledVideo(
            path=out_path, duration_seconds=0.0, width=width, height=height
        )
