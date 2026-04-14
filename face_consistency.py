"""
face_consistency.py - Face detection, embedding extraction, and character
consistency verification across video frames.

Uses MediaPipe for lightweight detection and optional InsightFace for
deep-feature embeddings.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class FaceDetection:
    """A detected face bounding box + optional embedding."""
    bbox: Tuple[int, int, int, int]   # (x, y, w, h)
    confidence: float
    embedding: Optional[np.ndarray] = None
    landmarks: Optional[np.ndarray] = None


@dataclass
class CharacterProfile:
    """Tracks a character's identity across frames."""
    character_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    reference_embedding: Optional[np.ndarray] = None
    sample_count: int = 0


# ---------------------------------------------------------------------------
# FaceConsistencyEngine
# ---------------------------------------------------------------------------

class FaceConsistencyEngine:
    """Detect faces, extract embeddings, and verify character consistency.

    Attempts to use InsightFace for deep embeddings; falls back to
    MediaPipe when InsightFace is unavailable.

    Args:
        similarity_threshold: Cosine similarity threshold above which
            two face embeddings are considered the same identity.
    """

    def __init__(self, similarity_threshold: float = 0.6) -> None:
        self._threshold = similarity_threshold
        self._mp_face: Any = None
        self._insight: Any = None
        self._init_backends()

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def _init_backends(self) -> None:
        # Try MediaPipe first (lighter dependency).
        try:
            import mediapipe as mp  # type: ignore

            self._mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
            log.info("MediaPipe face detection initialised.")
        except ImportError:
            log.warning("mediapipe not installed – face detection in mock mode.")

        # Try InsightFace for richer embeddings.
        try:
            import insightface  # type: ignore
            from insightface.app import FaceAnalysis  # type: ignore

            app = FaceAnalysis(providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            self._insight = app
            log.info("InsightFace initialised.")
        except ImportError:
            log.debug("insightface not available – using MediaPipe only.")
        except Exception as exc:  # pylint: disable=broad-except
            log.warning("InsightFace init failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Detection                                                            #
    # ------------------------------------------------------------------ #

    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces in an RGB image array.

        Args:
            image: HxWx3 uint8 numpy array (RGB).

        Returns:
            List of :class:`FaceDetection` objects.
        """
        if self._insight is not None:
            return self._detect_with_insightface(image)
        if self._mp_face is not None:
            return self._detect_with_mediapipe(image)
        return self._mock_detect(image)

    def detect_faces_in_frame(self, frame_path: Path) -> List[FaceDetection]:
        """Convenience wrapper that loads an image file then runs detection.

        Args:
            frame_path: Path to an image file.

        Returns:
            List of :class:`FaceDetection` objects.
        """
        try:
            import cv2  # type: ignore

            img = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)
            return self.detect_faces(img)
        except Exception as exc:  # pylint: disable=broad-except
            log.error("Could not load frame %s: %s", frame_path, exc)
            return []

    # ------------------------------------------------------------------ #
    # Embedding & Similarity                                               #
    # ------------------------------------------------------------------ #

    def get_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract the dominant face embedding from an image.

        Args:
            image: HxWx3 uint8 numpy array (RGB).

        Returns:
            1-D float32 embedding vector, or ``None`` if no face found.
        """
        faces = self.detect_faces(image)
        if not faces:
            return None
        # Return the embedding of the highest-confidence detection.
        best = max(faces, key=lambda f: f.confidence)
        return best.embedding

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embedding vectors.

        Args:
            a: First embedding.
            b: Second embedding.

        Returns:
            Similarity score in [-1, 1].
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def is_same_person(
        self, emb_a: np.ndarray, emb_b: np.ndarray
    ) -> bool:
        """Return ``True`` if two embeddings belong to the same person.

        Args:
            emb_a: First face embedding.
            emb_b: Second face embedding.

        Returns:
            Boolean identity verdict.
        """
        return self.cosine_similarity(emb_a, emb_b) >= self._threshold

    # ------------------------------------------------------------------ #
    # Character Profiles                                                   #
    # ------------------------------------------------------------------ #

    def build_character_profile(
        self, reference_images: List[np.ndarray], name: str = ""
    ) -> CharacterProfile:
        """Build a :class:`CharacterProfile` from multiple reference images.

        Averages embeddings across all detected faces to create a stable
        identity vector.

        Args:
            reference_images: List of RGB uint8 arrays containing the
                character's face.
            name: Optional human-readable name.

        Returns:
            :class:`CharacterProfile` instance.
        """
        embeddings: List[np.ndarray] = []
        for img in reference_images:
            emb = self.get_face_embedding(img)
            if emb is not None:
                embeddings.append(emb)

        profile = CharacterProfile(name=name, sample_count=len(embeddings))
        if embeddings:
            stacked = np.stack(embeddings, axis=0)
            profile.reference_embedding = stacked.mean(axis=0)
        return profile

    def verify_consistency(
        self,
        frames: List[np.ndarray],
        profile: CharacterProfile,
    ) -> Dict[str, Any]:
        """Check that *profile*'s identity appears consistently across *frames*.

        Args:
            frames: List of RGB frame arrays to check.
            profile: Reference character profile.

        Returns:
            Dict with ``consistent``, ``match_rate``, and ``scores``.
        """
        if profile.reference_embedding is None:
            return {"consistent": False, "match_rate": 0.0, "scores": []}

        scores: List[float] = []
        for frame in frames:
            emb = self.get_face_embedding(frame)
            if emb is not None:
                scores.append(
                    self.cosine_similarity(profile.reference_embedding, emb)
                )

        if not scores:
            return {"consistent": False, "match_rate": 0.0, "scores": []}

        match_rate = sum(s >= self._threshold for s in scores) / len(scores)
        return {
            "consistent": match_rate >= 0.8,
            "match_rate": round(match_rate, 3),
            "scores": [round(s, 3) for s in scores],
        }

    # ------------------------------------------------------------------ #
    # Internal Backends                                                    #
    # ------------------------------------------------------------------ #

    def _detect_with_insightface(self, image: np.ndarray) -> List[FaceDetection]:
        try:
            import cv2  # type: ignore

            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            faces_raw = self._insight.get(bgr)
            detections: List[FaceDetection] = []
            for f in faces_raw:
                box = f.bbox.astype(int)
                x1, y1, x2, y2 = box
                detections.append(
                    FaceDetection(
                        bbox=(x1, y1, x2 - x1, y2 - y1),
                        confidence=float(f.det_score),
                        embedding=f.embedding,
                        landmarks=f.landmark_2d_106 if hasattr(f, "landmark_2d_106") else None,
                    )
                )
            return detections
        except Exception as exc:  # pylint: disable=broad-except
            log.error("InsightFace detection error: %s", exc)
            return []

    def _detect_with_mediapipe(self, image: np.ndarray) -> List[FaceDetection]:
        try:
            results = self._mp_face.process(image)
            if not results.detections:
                return []
            h, w = image.shape[:2]
            detections: List[FaceDetection] = []
            for det in results.detections:
                bb = det.location_data.relative_bounding_box
                x = int(bb.xmin * w)
                y = int(bb.ymin * h)
                bw = int(bb.width * w)
                bh = int(bb.height * h)
                # MediaPipe doesn't provide embeddings – use a zero vector.
                detections.append(
                    FaceDetection(
                        bbox=(x, y, bw, bh),
                        confidence=det.score[0],
                        embedding=np.zeros(128, dtype=np.float32),
                    )
                )
            return detections
        except Exception as exc:  # pylint: disable=broad-except
            log.error("MediaPipe detection error: %s", exc)
            return []

    def _mock_detect(self, image: np.ndarray) -> List[FaceDetection]:
        """Return a single synthetic detection for testing."""
        h, w = image.shape[:2]
        return [
            FaceDetection(
                bbox=(w // 4, h // 4, w // 2, h // 2),
                confidence=0.9,
                embedding=np.random.randn(128).astype(np.float32),
            )
        ]
