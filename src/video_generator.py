"""Video generation stub.

This module provides a VideoGenerator class that submits AI-generated
prompts to a video generation back-end (e.g. Google Veo or a compatible
REST endpoint).  The actual HTTP call is isolated here so that swapping
providers only requires changing this file.
"""

import os
import time
import logging
import requests

logger = logging.getLogger(__name__)


class VideoGenerationError(Exception):
    """Raised when a video generation request fails."""


class VideoGenerator:
    """Submit video prompts to a generation API and save the result."""

    DEFAULT_POLL_INTERVAL = 5   # seconds between status checks
    DEFAULT_TIMEOUT = 300       # maximum seconds to wait for a video

    def __init__(
        self,
        api_key: str | None = None,
        output_dir: str = "output",
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key or os.environ.get("VEO_API_KEY", "")
        self.output_dir = output_dir
        self.poll_interval = poll_interval
        self.timeout = timeout
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(self, prompt: str, filename: str) -> str:
        """Generate a video from *prompt* and save it to *output_dir/filename*.

        Returns the full path of the saved video file.
        Raises :class:`VideoGenerationError` on failure.
        """
        output_path = os.path.join(self.output_dir, filename)

        # If no API key is configured, write a placeholder so the pipeline
        # can be tested end-to-end without a real credential.
        if not self.api_key:
            logger.warning(
                "VEO_API_KEY is not set. Writing placeholder file: %s", output_path
            )
            with open(output_path, "w", encoding="utf-8") as fh:
                fh.write(f"[placeholder] prompt: {prompt}\n")
            return output_path

        job_id = self._submit(prompt)
        video_url = self._poll(job_id)
        self._download(video_url, output_path)
        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _submit(self, prompt: str) -> str:
        """Submit a generation job and return its job ID."""
        try:
            response = requests.post(
                "https://api.veo.google.com/v1/videos",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"prompt": prompt},
                timeout=30,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = exc.response.text if exc.response is not None else ""
            raise VideoGenerationError(
                f"Failed to submit video job (HTTP {exc.response.status_code}): {body}"
            ) from exc
        return response.json()["job_id"]

    def _poll(self, job_id: str) -> str:
        """Poll until the job is complete and return the video URL."""
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                response = requests.get(
                    f"https://api.veo.google.com/v1/videos/{job_id}",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=30,
                )
                response.raise_for_status()
            except requests.HTTPError as exc:
                body = exc.response.text if exc.response is not None else ""
                raise VideoGenerationError(
                    f"Polling job {job_id} failed "
                    f"(HTTP {exc.response.status_code}): {body}"
                ) from exc
            data = response.json()
            status = data.get("status")
            if status == "completed":
                return data["video_url"]
            if status == "failed":
                raise VideoGenerationError(
                    f"Video generation failed for job {job_id}: {data.get('error')}"
                )
            logger.debug("Job %s status: %s — waiting…", job_id, status)
            time.sleep(self.poll_interval)
        raise VideoGenerationError(
            f"Timed out waiting for job {job_id} after {self.timeout}s"
        )

    def _download(self, url: str, path: str) -> None:
        """Stream-download *url* and save it to *path*."""
        try:
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with open(path, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=8192):
                        fh.write(chunk)
        except requests.HTTPError as exc:
            body = exc.response.text if exc.response is not None else ""
            raise VideoGenerationError(
                f"Failed to download video from {url} "
                f"(HTTP {exc.response.status_code}): {body}"
            ) from exc
        logger.info("Saved video to %s", path)
