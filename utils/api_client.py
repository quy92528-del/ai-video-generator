"""
utils/api_client.py - Simplified API wrapper for AI services (Gemini, Veo, Grok).

This module provides a lightweight :class:`APIClient` facade used by ``cli.py``
and other utilities.  For the full production client with key-rotation, retry
logic, and rate-limit handling see the root-level ``api_client.py``.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from utils.logger import get_logger

log = get_logger(__name__)


class APIClient:
    """Lightweight client for multiple AI services.

    Reads API keys from environment variables and lazily initializes each
    service on first use.
    """

    def __init__(self) -> None:
        """Initialize API Client from environment variables."""
        self.gemini_api_key: str = os.getenv("GOOGLE_GEMINI_API_KEY", "")
        self.veo_api_key: str = os.getenv("VEO_API_KEY", "")
        self.grok_api_key: str = os.getenv("GROK_API_KEY", "")

        self.gemini_model = None
        if self.gemini_api_key:
            try:
                import google.generativeai as genai  # type: ignore

                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel("gemini-1.5-pro")
                log.info("✓ Gemini API initialized")
            except Exception as exc:  # pylint: disable=broad-except
                log.warning(f"Could not initialize Gemini: {exc}")
        else:
            log.warning("GOOGLE_GEMINI_API_KEY not set – Gemini unavailable")

        log.info("APIClient ready")

    # ------------------------------------------------------------------
    # Text / Script Generation
    # ------------------------------------------------------------------

    def generate_text(self, prompt: str) -> Optional[str]:
        """Generate text using the Gemini API.

        Args:
            prompt: Full prompt string.

        Returns:
            Generated text, or ``None`` on error.
        """
        if not self.gemini_model:
            log.error("Gemini model not initialized")
            return None
        try:
            log.info(f"Generating text for prompt: {prompt[:60]}…")
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as exc:  # pylint: disable=broad-except
            log.error(f"Text generation failed: {exc}")
            return None

    def generate_script(self, title: str, duration: int = 60) -> Optional[str]:
        """Generate a video script for *title*.

        Args:
            title: Video title / topic.
            duration: Target duration in seconds.

        Returns:
            Script text, or ``None`` on error.
        """
        prompt = f"Write a {duration}-second video script about: {title}"
        return self.generate_text(prompt)

    # ------------------------------------------------------------------
    # Video Generation (Veo)
    # ------------------------------------------------------------------

    def generate_video(self, prompt: str, duration: int = 5) -> Optional[str]:
        """Generate a short video clip using the Veo API.

        Args:
            prompt: Visual description for the clip.
            duration: Desired clip length in seconds.

        Returns:
            Path to the generated video file, or ``None`` on error.
        """
        if not self.veo_api_key:
            log.error("VEO_API_KEY not set – Veo unavailable")
            return None
        try:
            log.info(f"Requesting Veo video ({duration}s): {prompt[:60]}…")
            # TODO: Implement Veo REST call when SDK is available.
            log.warning("Veo integration is not yet implemented")
            return None
        except Exception as exc:  # pylint: disable=broad-except
            log.error(f"Video generation failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Status / Health
    # ------------------------------------------------------------------

    def check_status(self) -> Dict[str, bool]:
        """Return availability status for each service.

        Returns:
            Mapping of service name → ``True`` if configured.
        """
        status: Dict[str, bool] = {
            "gemini": self.gemini_model is not None,
            "veo": bool(self.veo_api_key),
            "grok": bool(self.grok_api_key),
        }
        log.info(f"API status: {status}")
        return status

    def available_services(self) -> List[str]:
        """Return a list of service names that have valid credentials.

        Returns:
            List of available service name strings.
        """
        available = [name for name, ok in self.check_status().items() if ok]
        log.info(f"Available services: {available}")
        return available

    def test_connection(self) -> bool:
        """Run a lightweight connectivity check.

        Returns:
            ``True`` if at least one service responds successfully.
        """
        if self.gemini_model:
            result = self.generate_text("Respond with the single word: OK")
            if result:
                log.info("✓ Gemini connection test passed")
                return True
        log.warning("No services available for connection test")
        return False
