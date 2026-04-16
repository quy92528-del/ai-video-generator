"""
api_client.py - Unified API client with key rotation, retry logic, and
error handling across all external services.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

import httpx
from tenacity import (  # type: ignore
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from utils.logger import get_logger

log = get_logger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class APIError(Exception):
    """General API error."""


class RateLimitError(APIError):
    """Raised when a provider returns a 429 / rate-limit response."""


class AuthenticationError(APIError):
    """Raised when an API key is rejected (401 / 403)."""


class QuotaExceededError(APIError):
    """Raised when the account quota is exhausted."""


# ---------------------------------------------------------------------------
# Key Rotation Pool
# ---------------------------------------------------------------------------

@dataclass
class APIKeyPool:
    """Round-robin API key pool with automatic rotation on failure.

    Args:
        keys: List of API key strings.
    """

    keys: List[str]
    _index: int = field(default=0, init=False, repr=False)
    _failed: List[str] = field(default_factory=list, init=False, repr=False)

    def current(self) -> str:
        """Return the active API key."""
        available = self._available()
        if not available:
            raise AuthenticationError("All API keys in pool have failed.")
        return available[self._index % len(available)]

    def rotate(self) -> str:
        """Mark the current key as failed and advance to the next."""
        bad = self.current()
        if bad not in self._failed:
            log.warning("Rotating away from failed API key: …%s", bad[-6:])
            self._failed.append(bad)
        self._index += 1
        return self.current()

    def _available(self) -> List[str]:
        return [k for k in self.keys if k not in self._failed]

    @property
    def has_keys(self) -> bool:
        return bool(self._available())


# ---------------------------------------------------------------------------
# Unified HTTP Client
# ---------------------------------------------------------------------------

class UnifiedAPIClient:
    """HTTP client with retry, key rotation, and rate-limit back-off.

    Args:
        timeout: Default request timeout in seconds.
        max_retries: Maximum retry attempts per request.
        retry_delay: Base retry delay in seconds (exponential back-off).
    """

    def __init__(
        self,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._client = httpx.Client(timeout=timeout)

    def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        key_pool: Optional[APIKeyPool] = None,
        auth_header: str = "Authorization",
        auth_prefix: str = "Bearer",
    ) -> httpx.Response:
        """Perform a GET request with retry and optional key rotation.

        Args:
            url: Endpoint URL.
            headers: Additional HTTP headers.
            params: URL query parameters.
            key_pool: Optional key pool for rotation on 401/429.
            auth_header: Header name for API key auth.
            auth_prefix: Header value prefix (e.g. ``"Bearer"``).

        Returns:
            :class:`httpx.Response`.

        Raises:
            APIError: On permanent failure after exhausting retries.
        """
        return self._request(
            "GET", url, headers, params=params, key_pool=key_pool,
            auth_header=auth_header, auth_prefix=auth_prefix,
        )

    def post(
        self,
        url: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
        key_pool: Optional[APIKeyPool] = None,
        auth_header: str = "Authorization",
        auth_prefix: str = "Bearer",
    ) -> httpx.Response:
        """Perform a POST request with retry and optional key rotation.

        Args:
            url: Endpoint URL.
            json: Request body as a JSON-serialisable dict.
            data: Raw request body bytes.
            headers: Additional HTTP headers.
            key_pool: Optional key pool for rotation on 401/429.
            auth_header: Header name for API key auth.
            auth_prefix: Header value prefix.

        Returns:
            :class:`httpx.Response`.

        Raises:
            APIError: On permanent failure after exhausting retries.
        """
        return self._request(
            "POST", url, headers, json=json, content=data, key_pool=key_pool,
            auth_header=auth_header, auth_prefix=auth_prefix,
        )

    def _request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]],
        json: Optional[Dict[str, Any]] = None,
        content: Optional[bytes] = None,
        params: Optional[Dict[str, Any]] = None,
        key_pool: Optional[APIKeyPool] = None,
        auth_header: str = "Authorization",
        auth_prefix: str = "Bearer",
    ) -> httpx.Response:
        h = dict(headers or {})
        attempt = 0
        delay = self._retry_delay

        while attempt < self._max_retries:
            if key_pool and key_pool.has_keys:
                h[auth_header] = f"{auth_prefix} {key_pool.current()}"
            try:
                resp = self._client.request(
                    method,
                    url,
                    headers=h,
                    json=json,
                    content=content,
                    params=params,
                )
                if resp.status_code == 200:
                    return resp
                if resp.status_code == 429:
                    wait = self._parse_retry_after(resp) or delay
                    log.warning(
                        "Rate limited (429) – waiting %.0fs before retry.", wait
                    )
                    time.sleep(wait)
                    delay *= 2
                    attempt += 1
                    continue
                if resp.status_code in (401, 403):
                    if key_pool and key_pool.has_keys:
                        key_pool.rotate()
                        attempt += 1
                        continue
                    raise AuthenticationError(
                        f"Authentication failed for {url}: HTTP {resp.status_code}"
                    )
                resp.raise_for_status()
                return resp
            except httpx.TimeoutException as exc:
                log.warning("Request timeout (attempt %d): %s", attempt + 1, exc)
            except httpx.RequestError as exc:
                log.warning("Request error (attempt %d): %s", attempt + 1, exc)

            time.sleep(delay)
            delay = min(delay * 2, 60.0)
            attempt += 1

        raise APIError(
            f"Request to {url} failed after {self._max_retries} attempts."
        )

    @staticmethod
    def _parse_retry_after(response: httpx.Response) -> Optional[float]:
        """Parse the Retry-After header if present."""
        val = response.headers.get("Retry-After")
        if val is None:
            return None
        try:
            return float(val)
        except ValueError:
            return None

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> "UnifiedAPIClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Service-specific Thin Wrappers
# ---------------------------------------------------------------------------

class GeminiClient:
    """Thin wrapper around the Gemini generative AI REST API."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_keys: List[str], model: str = "gemini-1.5-pro") -> None:
        self._pool = APIKeyPool(keys=api_keys)
        self._model = model
        self._http = UnifiedAPIClient()

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Send a prompt and return the text response.

        Args:
            prompt: Input prompt string.
            temperature: Sampling temperature.

        Returns:
            Generated text string.
        """
        url = f"{self.BASE_URL}/{self._model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        params = {"key": self._pool.current()}
        try:
            resp = self._http.post(url, json=payload, params=params)
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as exc:  # pylint: disable=broad-except
            log.error("Gemini generate error: %s", exc)
            return ""


class RunwayClient:
    """Thin wrapper for the Runway Gen-3 API."""

    BASE = "https://api.dev.runwayml.com/v1"

    def __init__(self, api_key: str, version: str = "2024-11-06") -> None:
        self._key = api_key
        self._version = version
        self._http = UnifiedAPIClient(timeout=300)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._key}",
            "X-Runway-Version": self._version,
            "Content-Type": "application/json",
        }

    def create_image_to_video_task(
        self, image_b64: str, prompt: str, duration: int = 5
    ) -> str:
        """Submit an image-to-video task and return the task ID."""
        payload = {
            "model": "gen3a_turbo",
            "promptImage": f"data:image/png;base64,{image_b64}",
            "promptText": prompt,
            "duration": duration,
        }
        resp = self._http.post(
            f"{self.BASE}/image_to_video", json=payload, headers=self._headers()
        )
        return resp.json()["id"]

    def get_task(self, task_id: str) -> Dict[str, Any]:
        """Return the current state of a task."""
        resp = self._http.get(
            f"{self.BASE}/tasks/{task_id}", headers=self._headers()
        )
        return resp.json()
