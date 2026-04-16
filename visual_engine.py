"""
visual_engine.py - Image & video generation.

Integrates Stable Diffusion (via Replicate), Runway Gen-3, and HeyGen
avatar creation.  Provides a unified interface with automatic fallback
between providers.
"""

from __future__ import annotations

import base64
import os
import time
import urllib.request
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
class GeneratedImage:
    """Result of an image-generation request."""
    path: Path
    prompt: str
    width: int
    height: int
    provider: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedVideo:
    """Result of a video-generation request."""
    path: Path
    prompt: str
    duration_seconds: float
    width: int
    height: int
    provider: str
    fps: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# VisualEngine
# ---------------------------------------------------------------------------

class VisualEngine:
    """Unified image & video generation with provider fallback.

    Args:
        replicate_token: Replicate API token (Stable Diffusion, Wav2Lip).
        runway_api_key: Runway ML API key (Gen-3 video).
        heygen_api_key: HeyGen API key (avatar videos).
        output_dir: Directory to save generated files.
        sd_model: Replicate model identifier for Stable Diffusion.
    """

    def __init__(
        self,
        replicate_token: str = "",
        runway_api_key: str = "",
        heygen_api_key: str = "",
        output_dir: str = "output",
        sd_model: str = "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
    ) -> None:
        self._replicate_token = replicate_token
        self._runway_key = runway_api_key
        self._heygen_key = heygen_api_key
        self._output_dir = Path(output_dir)
        self._sd_model = sd_model
        self._replicate: Any = None
        self._init_replicate()

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def _init_replicate(self) -> None:
        if not self._replicate_token:
            log.warning("Replicate token not set – image generation in mock mode.")
            return
        try:
            import replicate as repl  # type: ignore

            os.environ.setdefault("REPLICATE_API_TOKEN", self._replicate_token)
            self._replicate = repl
            log.info("Replicate client initialised.")
        except ImportError:
            log.warning("replicate package not installed – running in mock mode.")

    # ------------------------------------------------------------------ #
    # Image Generation                                                     #
    # ------------------------------------------------------------------ #

    def generate_image(
        self,
        prompt: str,
        width: int = 1080,
        height: int = 1920,
        negative_prompt: str = "blurry, low quality, distorted, watermark",
        style_preset: str = "cinematic",
        output_dir: Optional[str] = None,
    ) -> GeneratedImage:
        """Generate a single image from *prompt*.

        Tries Replicate (Stable Diffusion SDXL) first, falls back to a mock
        placeholder when unavailable.

        Args:
            prompt: Positive image prompt.
            width: Output image width in pixels.
            height: Output image height in pixels.
            negative_prompt: Negative conditioning prompt.
            style_preset: Visual style hint added to the prompt.
            output_dir: Override default output directory.

        Returns:
            :class:`GeneratedImage` with the saved file path.
        """
        save_dir = ensure_dir(output_dir or self._output_dir / "images")

        if self._replicate is not None:
            return self._generate_with_replicate(
                prompt, width, height, negative_prompt, style_preset, save_dir
            )
        return self._generate_mock_image(prompt, width, height, save_dir)

    def generate_images_batch(
        self,
        prompts: List[str],
        width: int = 1080,
        height: int = 1920,
        output_dir: Optional[str] = None,
    ) -> List[GeneratedImage]:
        """Generate multiple images (one per prompt).

        Args:
            prompts: List of prompt strings.
            width: Image width.
            height: Image height.
            output_dir: Override output directory.

        Returns:
            List of :class:`GeneratedImage` results.
        """
        results: List[GeneratedImage] = []
        for i, prompt in enumerate(prompts):
            log.info("Generating image %d/%d …", i + 1, len(prompts))
            try:
                img = self.generate_image(prompt, width, height, output_dir=output_dir)
                results.append(img)
            except Exception as exc:  # pylint: disable=broad-except
                log.error("Image generation failed for prompt %d: %s", i, exc)
            # Brief sleep to avoid hammering the API.
            if i < len(prompts) - 1:
                time.sleep(0.5)
        return results

    # ------------------------------------------------------------------ #
    # Video Generation                                                     #
    # ------------------------------------------------------------------ #

    def generate_video_from_image(
        self,
        image_path: Path,
        prompt: str,
        duration_seconds: int = 5,
        output_dir: Optional[str] = None,
    ) -> GeneratedVideo:
        """Animate a static image into a short video clip (Runway Gen-3).

        Args:
            image_path: Path to the input image.
            prompt: Motion prompt describing how the scene should move.
            duration_seconds: Clip length (5 or 10 seconds for Runway).
            output_dir: Override output directory.

        Returns:
            :class:`GeneratedVideo` with the saved file path.
        """
        save_dir = ensure_dir(output_dir or self._output_dir / "clips")

        if self._runway_key:
            return self._generate_with_runway(
                image_path, prompt, duration_seconds, save_dir
            )
        return self._generate_mock_video(prompt, duration_seconds, save_dir)

    def generate_cinematic_video(
        self,
        prompt: str,
        duration_seconds: int = 10,
        width: int = 1080,
        height: int = 1920,
        output_dir: Optional[str] = None,
    ) -> GeneratedVideo:
        """Generate a text-to-video clip (Runway Gen-3 text-to-video).

        Args:
            prompt: Scene description for video generation.
            duration_seconds: Target duration.
            width: Output width.
            height: Output height.
            output_dir: Override output directory.

        Returns:
            :class:`GeneratedVideo`.
        """
        save_dir = ensure_dir(output_dir or self._output_dir / "clips")
        if self._runway_key:
            return self._runway_text_to_video(
                prompt, duration_seconds, width, height, save_dir
            )
        return self._generate_mock_video(prompt, duration_seconds, save_dir)

    def create_avatar_video(
        self,
        script_text: str,
        avatar_id: str = "default",
        audio_path: Optional[Path] = None,
        output_dir: Optional[str] = None,
    ) -> GeneratedVideo:
        """Create an avatar presenter video via HeyGen.

        Args:
            script_text: The script the avatar will speak.
            avatar_id: HeyGen avatar ID.
            audio_path: Optional pre-generated audio file for lip-sync.
            output_dir: Override output directory.

        Returns:
            :class:`GeneratedVideo`.
        """
        save_dir = ensure_dir(output_dir or self._output_dir / "clips")
        if self._heygen_key:
            return self._generate_with_heygen(
                script_text, avatar_id, audio_path, save_dir
            )
        return self._generate_mock_video(script_text, 30, save_dir)

    # ------------------------------------------------------------------ #
    # Provider: Replicate (Stable Diffusion)                              #
    # ------------------------------------------------------------------ #

    def _generate_with_replicate(
        self,
        prompt: str,
        width: int,
        height: int,
        negative_prompt: str,
        style_preset: str,
        save_dir: Path,
    ) -> GeneratedImage:
        full_prompt = f"{style_preset} style, {prompt}, high quality, 4K, detailed"
        log.debug("Replicate SD request: prompt='%s'", full_prompt[:80])

        try:
            output = self._replicate.run(
                self._sd_model,
                input={
                    "prompt": full_prompt,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5,
                },
            )
            url = output[0] if isinstance(output, list) else str(output)
            path = unique_filename(save_dir, "image", ".png")
            self._download_file(url, path)
            log.info("Image saved: %s", path)
            return GeneratedImage(
                path=path,
                prompt=full_prompt,
                width=width,
                height=height,
                provider="replicate",
            )
        except Exception as exc:  # pylint: disable=broad-except
            log.error("Replicate error: %s – falling back to mock.", exc)
            return self._generate_mock_image(prompt, width, height, save_dir)

    # ------------------------------------------------------------------ #
    # Provider: Runway Gen-3                                              #
    # ------------------------------------------------------------------ #

    def _generate_with_runway(
        self,
        image_path: Path,
        prompt: str,
        duration_seconds: int,
        save_dir: Path,
    ) -> GeneratedVideo:
        """Image-to-video via Runway Gen-3 Alpha API."""
        import httpx  # type: ignore

        endpoint = "https://api.dev.runwayml.com/v1/image_to_video"
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        headers = {
            "Authorization": f"Bearer {self._runway_key}",
            "X-Runway-Version": "2024-11-06",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gen3a_turbo",
            "promptImage": f"data:image/png;base64,{img_b64}",
            "promptText": prompt,
            "duration": min(duration_seconds, 10),
            "ratio": "768:1280",
        }

        try:
            resp = httpx.post(endpoint, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            task_id = resp.json()["id"]
            video_url = self._poll_runway_task(task_id, headers)
            path = unique_filename(save_dir, "clip", ".mp4")
            self._download_file(video_url, path)
            log.info("Runway video saved: %s", path)
            return GeneratedVideo(
                path=path,
                prompt=prompt,
                duration_seconds=duration_seconds,
                width=768,
                height=1280,
                provider="runway",
            )
        except Exception as exc:  # pylint: disable=broad-except
            log.error("Runway error: %s – falling back to mock.", exc)
            return self._generate_mock_video(prompt, duration_seconds, save_dir)

    def _runway_text_to_video(
        self,
        prompt: str,
        duration_seconds: int,
        width: int,
        height: int,
        save_dir: Path,
    ) -> GeneratedVideo:
        """Text-to-video via Runway Gen-3."""
        import httpx  # type: ignore

        endpoint = "https://api.dev.runwayml.com/v1/text_to_video"
        headers = {
            "Authorization": f"Bearer {self._runway_key}",
            "X-Runway-Version": "2024-11-06",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gen3a_turbo",
            "promptText": prompt,
            "duration": min(duration_seconds, 10),
        }
        try:
            resp = httpx.post(endpoint, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            task_id = resp.json()["id"]
            video_url = self._poll_runway_task(task_id, headers)
            path = unique_filename(save_dir, "clip", ".mp4")
            self._download_file(video_url, path)
            return GeneratedVideo(
                path=path,
                prompt=prompt,
                duration_seconds=duration_seconds,
                width=width,
                height=height,
                provider="runway",
            )
        except Exception as exc:  # pylint: disable=broad-except
            log.error("Runway t2v error: %s – falling back to mock.", exc)
            return self._generate_mock_video(prompt, duration_seconds, save_dir)

    def _poll_runway_task(
        self,
        task_id: str,
        headers: Dict[str, str],
        max_wait: int = 300,
        interval: int = 5,
    ) -> str:
        """Poll a Runway task until complete and return the output URL."""
        import httpx  # type: ignore

        url = f"https://api.dev.runwayml.com/v1/tasks/{task_id}"
        elapsed = 0
        while elapsed < max_wait:
            resp = httpx.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status")
            if status == "SUCCEEDED":
                return data["output"][0]
            if status in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"Runway task {task_id} ended with status {status}")
            time.sleep(interval)
            elapsed += interval
        raise TimeoutError(f"Runway task {task_id} timed out after {max_wait}s.")

    # ------------------------------------------------------------------ #
    # Provider: HeyGen                                                     #
    # ------------------------------------------------------------------ #

    def _generate_with_heygen(
        self,
        script_text: str,
        avatar_id: str,
        audio_path: Optional[Path],
        save_dir: Path,
    ) -> GeneratedVideo:
        """Create an avatar video via HeyGen v2 API."""
        import httpx  # type: ignore

        headers = {
            "X-Api-Key": self._heygen_key,
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "video_inputs": [
                {
                    "character": {
                        "type": "avatar",
                        "avatar_id": avatar_id,
                        "avatar_style": "normal",
                    },
                    "voice": {
                        "type": "text",
                        "input_text": script_text,
                        "voice_id": "2d5b0e6cf36f460aa7fc47e3eee4ba54",
                    },
                }
            ],
            "dimension": {"width": 1080, "height": 1920},
        }
        try:
            resp = httpx.post(
                "https://api.heygen.com/v2/video/generate",
                json=payload,
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            video_id = resp.json()["data"]["video_id"]
            video_url = self._poll_heygen_video(video_id, headers)
            path = unique_filename(save_dir, "avatar", ".mp4")
            self._download_file(video_url, path)
            log.info("HeyGen avatar video saved: %s", path)
            return GeneratedVideo(
                path=path,
                prompt=script_text[:100],
                duration_seconds=0,
                width=1080,
                height=1920,
                provider="heygen",
            )
        except Exception as exc:  # pylint: disable=broad-except
            log.error("HeyGen error: %s – falling back to mock.", exc)
            return self._generate_mock_video(script_text, 30, save_dir)

    def _poll_heygen_video(
        self,
        video_id: str,
        headers: Dict[str, str],
        max_wait: int = 600,
        interval: int = 10,
    ) -> str:
        import httpx  # type: ignore

        url = f"https://api.heygen.com/v1/video_status.get?video_id={video_id}"
        elapsed = 0
        while elapsed < max_wait:
            resp = httpx.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()["data"]
            if data["status"] == "completed":
                return data["video_url"]
            if data["status"] == "failed":
                raise RuntimeError(f"HeyGen video {video_id} failed.")
            time.sleep(interval)
            elapsed += interval
        raise TimeoutError(f"HeyGen video {video_id} timed out.")

    # ------------------------------------------------------------------ #
    # Mock / Fallback                                                      #
    # ------------------------------------------------------------------ #

    def _generate_mock_image(
        self, prompt: str, width: int, height: int, save_dir: Path
    ) -> GeneratedImage:
        """Generate a placeholder gradient image using Pillow."""
        try:
            from PIL import Image, ImageDraw, ImageFont  # type: ignore
            import numpy as np  # type: ignore

            arr = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                arr[y, :, 0] = int(30 + 60 * y / height)
                arr[y, :, 2] = int(80 + 120 * (1 - y / height))

            img = Image.fromarray(arr, "RGB")
            draw = ImageDraw.Draw(img)
            text = prompt[:60]
            draw.text((10, 10), text, fill=(255, 255, 255))
        except ImportError:
            # PIL not available – write an empty file.
            from pathlib import Path as P

            path = unique_filename(save_dir, "mock_image", ".png")
            path.write_bytes(b"")
            return GeneratedImage(
                path=path, prompt=prompt, width=width, height=height, provider="mock"
            )

        path = unique_filename(save_dir, "mock_image", ".png")
        img.save(path)
        log.debug("Mock image saved: %s", path)
        return GeneratedImage(
            path=path, prompt=prompt, width=width, height=height, provider="mock"
        )

    def _generate_mock_video(
        self, prompt: str, duration_seconds: int, save_dir: Path
    ) -> GeneratedVideo:
        """Return a mock video (empty file) for testing without an API key."""
        path = unique_filename(save_dir, "mock_video", ".mp4")
        path.write_bytes(b"")
        log.debug("Mock video created: %s (prompt='%s')", path, prompt[:50])
        return GeneratedVideo(
            path=path,
            prompt=prompt,
            duration_seconds=float(duration_seconds),
            width=1080,
            height=1920,
            provider="mock",
        )

    # ------------------------------------------------------------------ #
    # Utility                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _download_file(url: str, dest: Path) -> None:
        """Download *url* to *dest*."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(dest))
