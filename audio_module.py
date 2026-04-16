"""
audio_module.py - TTS, voice cloning, audio mixing, and ducking.

Supports Google Cloud TTS, ElevenLabs, and a local pyttsx3 fallback.
Handles Wav2Lip lip-sync via Replicate.
"""

from __future__ import annotations

import os
import tempfile
import time
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
class AudioTrack:
    """A single audio track (voice-over, music, or SFX)."""
    path: Path
    duration_seconds: float
    language: str = "vi"
    provider: str = "mock"
    sample_rate: int = 44100
    channels: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MixedAudio:
    """Result of mixing multiple audio tracks."""
    path: Path
    duration_seconds: float
    sample_rate: int = 44100


# ---------------------------------------------------------------------------
# Language → TTS voice mapping
# ---------------------------------------------------------------------------

GOOGLE_TTS_VOICES: Dict[str, Tuple[str, str]] = {
    # (language_code, voice_name)
    "vi": ("vi-VN", "vi-VN-Neural2-A"),
    "en": ("en-US", "en-US-Neural2-F"),
    "zh": ("cmn-CN", "cmn-CN-Neural2-A"),
    "ja": ("ja-JP", "ja-JP-Neural2-B"),
    "ko": ("ko-KR", "ko-KR-Neural2-A"),
    "es": ("es-US", "es-US-Neural2-A"),
    "fr": ("fr-FR", "fr-FR-Neural2-A"),
}

ELEVENLABS_VOICE_IDS: Dict[str, str] = {
    "vi": "21m00Tcm4TlvDq8ikWAM",  # Rachel (multilingual)
    "en": "21m00Tcm4TlvDq8ikWAM",  # Rachel
    "zh": "AZnzlk1XvdvUeBnXmlld",  # Domi
    "ja": "EXAVITQu4vr4xnSDxMaL",  # Bella
    "ko": "21m00Tcm4TlvDq8ikWAM",  # Rachel (fallback)
    "es": "VR6AewLTigWG4xSOukaG",  # Arnold
    "fr": "pNInz6obpgDQGcFmaJgB",  # Adam
}


# ---------------------------------------------------------------------------
# AudioModule
# ---------------------------------------------------------------------------

class AudioModule:
    """TTS, voice cloning, Wav2Lip lip-sync, and audio mixing.

    Args:
        google_credentials_json: Path to Google Cloud service account JSON.
        elevenlabs_api_key: ElevenLabs API key.
        replicate_token: Replicate API token (for Wav2Lip).
        output_dir: Directory for generated audio files.
    """

    def __init__(
        self,
        google_credentials_json: str = "",
        elevenlabs_api_key: str = "",
        replicate_token: str = "",
        output_dir: str = "output",
    ) -> None:
        self._gcp_creds = google_credentials_json
        self._elevenlabs_key = elevenlabs_api_key
        self._replicate_token = replicate_token
        self._output_dir = Path(output_dir)
        self._replicate: Any = None
        self._init_replicate()

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def _init_replicate(self) -> None:
        if not self._replicate_token:
            return
        try:
            import replicate as repl  # type: ignore

            os.environ.setdefault("REPLICATE_API_TOKEN", self._replicate_token)
            self._replicate = repl
        except ImportError:
            log.warning("replicate package not installed – Wav2Lip unavailable.")

    # ------------------------------------------------------------------ #
    # TTS                                                                  #
    # ------------------------------------------------------------------ #

    def synthesize_speech(
        self,
        text: str,
        language: str = "vi",
        provider: str = "auto",
        output_dir: Optional[str] = None,
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
    ) -> AudioTrack:
        """Synthesise *text* into speech audio.

        Args:
            text: Text to synthesise.
            language: ISO language code.
            provider: ``"google"``, ``"elevenlabs"``, ``"pyttsx3"``,
                      or ``"auto"`` (tries Google → ElevenLabs → pyttsx3).
            output_dir: Override output directory.
            speaking_rate: Speech speed (0.5–2.0, default 1.0).
            pitch: Pitch adjustment in semitones (-10 to +10).

        Returns:
            :class:`AudioTrack` with the saved file path.
        """
        save_dir = ensure_dir(output_dir or self._output_dir / "audio")

        if provider == "auto":
            if self._gcp_creds or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                provider = "google"
            elif self._elevenlabs_key:
                provider = "elevenlabs"
            else:
                provider = "pyttsx3"

        if provider == "google":
            return self._google_tts(text, language, save_dir, speaking_rate, pitch)
        if provider == "elevenlabs":
            return self._elevenlabs_tts(text, language, save_dir)
        return self._pyttsx3_tts(text, language, save_dir)

    def synthesize_batch(
        self,
        texts: List[str],
        language: str = "vi",
        provider: str = "auto",
        output_dir: Optional[str] = None,
    ) -> List[AudioTrack]:
        """Synthesise a list of texts.

        Args:
            texts: List of strings to synthesise.
            language: Language code.
            provider: TTS provider.
            output_dir: Override output directory.

        Returns:
            List of :class:`AudioTrack` objects.
        """
        tracks: List[AudioTrack] = []
        for i, text in enumerate(texts):
            log.info("Synthesising audio %d/%d …", i + 1, len(texts))
            try:
                track = self.synthesize_speech(
                    text, language, provider, output_dir
                )
                tracks.append(track)
            except Exception as exc:  # pylint: disable=broad-except
                log.error("TTS failed for item %d: %s", i, exc)
                tracks.append(self._silent_track(save_dir=Path(output_dir or self._output_dir / "audio")))
            time.sleep(0.2)
        return tracks

    # ------------------------------------------------------------------ #
    # Wav2Lip Lip-sync                                                     #
    # ------------------------------------------------------------------ #

    def apply_lip_sync(
        self,
        video_path: Path,
        audio_path: Path,
        output_dir: Optional[str] = None,
    ) -> Path:
        """Apply Wav2Lip lip-sync to a video.

        Args:
            video_path: Input video file (avatar/portrait).
            audio_path: Input audio file (speech).
            output_dir: Override output directory.

        Returns:
            Path to the lip-synced video.
        """
        save_dir = ensure_dir(output_dir or self._output_dir / "lipsync")
        out_path = unique_filename(save_dir, "lipsync", ".mp4")

        if self._replicate is None:
            log.warning("Wav2Lip unavailable – copying input video as-is.")
            import shutil

            shutil.copy2(video_path, out_path)
            return out_path

        try:
            output = self._replicate.run(
                "devxpy/cog-wav2lip:8d65e3f4f4298519a9d84e2f40c4ee69f3c9a6c9cb7b88c93cc3c9da7b88c9c3",
                input={
                    "face": open(video_path, "rb"),
                    "audio": open(audio_path, "rb"),
                },
            )
            url = str(output)
            from utils.helpers import ensure_dir as _ed
            from visual_engine import VisualEngine

            VisualEngine._download_file(url, out_path)
            log.info("Lip-synced video saved: %s", out_path)
            return out_path
        except Exception as exc:  # pylint: disable=broad-except
            log.error("Wav2Lip error: %s", exc)
            import shutil

            shutil.copy2(video_path, out_path)
            return out_path

    # ------------------------------------------------------------------ #
    # Audio Mixing                                                         #
    # ------------------------------------------------------------------ #

    def mix_audio(
        self,
        voice_track: AudioTrack,
        music_path: Optional[Path] = None,
        sfx_paths: Optional[List[Path]] = None,
        music_volume: float = 0.15,
        sfx_volume: float = 0.6,
        output_dir: Optional[str] = None,
    ) -> MixedAudio:
        """Mix voice-over with background music and SFX.

        Applies audio ducking so the voice-over is prominent.

        Args:
            voice_track: Primary voice-over audio.
            music_path: Optional background music file.
            sfx_paths: Optional list of SFX files.
            music_volume: Background music volume (0.0–1.0).
            sfx_volume: SFX volume (0.0–1.0).
            output_dir: Override output directory.

        Returns:
            :class:`MixedAudio` with the mixed file path.
        """
        save_dir = ensure_dir(output_dir or self._output_dir / "mixed")
        out_path = unique_filename(save_dir, "mixed", ".mp3")

        try:
            from pydub import AudioSegment  # type: ignore

            voice = AudioSegment.from_file(str(voice_track.path))
            result = voice

            if music_path and music_path.exists():
                bg = AudioSegment.from_file(str(music_path))
                bg = bg - int((1 - music_volume) * 20)  # Attenuate
                # Loop or trim to voice length.
                if len(bg) < len(voice):
                    repeats = (len(voice) // len(bg)) + 2
                    bg = bg * repeats
                bg = bg[: len(voice)]
                result = bg.overlay(voice)

            if sfx_paths:
                for sfx_path in sfx_paths:
                    if sfx_path.exists():
                        sfx = AudioSegment.from_file(str(sfx_path))
                        sfx = sfx - int((1 - sfx_volume) * 20)
                        result = result.overlay(sfx)

            result.export(str(out_path), format="mp3")
            duration = len(result) / 1000.0
            log.info("Mixed audio saved: %s (%.1fs)", out_path, duration)
            return MixedAudio(path=out_path, duration_seconds=duration)

        except ImportError:
            log.warning("pydub not available – returning voice track as-is.")
            import shutil

            shutil.copy2(voice_track.path, out_path)
            return MixedAudio(
                path=out_path, duration_seconds=voice_track.duration_seconds
            )
        except Exception as exc:  # pylint: disable=broad-except
            log.error("Audio mixing failed: %s", exc)
            import shutil

            shutil.copy2(voice_track.path, out_path)
            return MixedAudio(
                path=out_path, duration_seconds=voice_track.duration_seconds
            )

    def get_audio_duration(self, path: Path) -> float:
        """Return duration of an audio file in seconds.

        Args:
            path: Path to audio file.

        Returns:
            Duration in seconds, or 0.0 on error.
        """
        try:
            from pydub import AudioSegment  # type: ignore

            audio = AudioSegment.from_file(str(path))
            return len(audio) / 1000.0
        except Exception:  # pylint: disable=broad-except
            return 0.0

    # ------------------------------------------------------------------ #
    # Provider: Google Cloud TTS                                           #
    # ------------------------------------------------------------------ #

    def _google_tts(
        self,
        text: str,
        language: str,
        save_dir: Path,
        speaking_rate: float,
        pitch: float,
    ) -> AudioTrack:
        try:
            from google.cloud import texttospeech  # type: ignore

            if self._gcp_creds:
                os.environ.setdefault(
                    "GOOGLE_APPLICATION_CREDENTIALS", self._gcp_creds
                )

            client = texttospeech.TextToSpeechClient()
            lang_code, voice_name = GOOGLE_TTS_VOICES.get(language, ("en-US", "en-US-Neural2-F"))
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code=lang_code,
                name=voice_name,
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=speaking_rate,
                pitch=pitch,
            )
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config,
            )
            path = unique_filename(save_dir, "tts_google", ".mp3")
            path.write_bytes(response.audio_content)
            duration = self.get_audio_duration(path)
            log.info("Google TTS saved: %s (%.1fs)", path, duration)
            return AudioTrack(
                path=path,
                duration_seconds=duration,
                language=language,
                provider="google",
            )
        except Exception as exc:  # pylint: disable=broad-except
            log.error("Google TTS error: %s – falling back to pyttsx3.", exc)
            return self._pyttsx3_tts(text, language, save_dir)

    # ------------------------------------------------------------------ #
    # Provider: ElevenLabs                                                 #
    # ------------------------------------------------------------------ #

    def _elevenlabs_tts(
        self, text: str, language: str, save_dir: Path
    ) -> AudioTrack:
        try:
            import httpx  # type: ignore

            voice_id = ELEVENLABS_VOICE_IDS.get(language, "21m00Tcm4TlvDq8ikWAM")
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "xi-api-key": self._elevenlabs_key,
                "Content-Type": "application/json",
            }
            payload = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            }
            resp = httpx.post(url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            path = unique_filename(save_dir, "tts_elevenlabs", ".mp3")
            path.write_bytes(resp.content)
            duration = self.get_audio_duration(path)
            log.info("ElevenLabs TTS saved: %s (%.1fs)", path, duration)
            return AudioTrack(
                path=path,
                duration_seconds=duration,
                language=language,
                provider="elevenlabs",
            )
        except Exception as exc:  # pylint: disable=broad-except
            log.error("ElevenLabs error: %s – falling back to pyttsx3.", exc)
            return self._pyttsx3_tts(text, language, save_dir)

    # ------------------------------------------------------------------ #
    # Provider: pyttsx3 (local fallback)                                   #
    # ------------------------------------------------------------------ #

    def _pyttsx3_tts(
        self, text: str, language: str, save_dir: Path  # noqa: ARG002
    ) -> AudioTrack:
        path = unique_filename(save_dir, "tts_local", ".mp3")
        try:
            import pyttsx3  # type: ignore

            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            import shutil

            shutil.move(tmp_path, str(path))
            duration = self.get_audio_duration(path)
            return AudioTrack(
                path=path,
                duration_seconds=duration,
                language=language,
                provider="pyttsx3",
            )
        except Exception as exc:  # pylint: disable=broad-except
            log.warning("pyttsx3 unavailable (%s) – creating silent track.", exc)
            return self._silent_track(save_dir=save_dir, path=path)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _silent_track(
        self,
        duration_ms: int = 3000,
        save_dir: Optional[Path] = None,
        path: Optional[Path] = None,
    ) -> AudioTrack:
        """Create a short silent audio track as a fallback."""
        if path is None:
            save_dir = save_dir or ensure_dir(self._output_dir / "audio")
            path = unique_filename(save_dir, "silent", ".mp3")
        try:
            from pydub import AudioSegment  # type: ignore
            from pydub.generators import Sine  # type: ignore  # noqa: F401

            silent = AudioSegment.silent(duration=duration_ms)
            silent.export(str(path), format="mp3")
        except Exception:  # pylint: disable=broad-except
            path.write_bytes(b"")  # empty file as last resort
        return AudioTrack(path=path, duration_seconds=duration_ms / 1000.0)
