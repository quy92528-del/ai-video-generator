"""
avatar_engine.py - Face/Avatar generation from reference images.

Provides face-consistent image, avatar, and video generation using a
user-supplied reference face image.  Supports multiple styles (realistic,
animated, 3D, Pixar) and batch processing for multiple faces.

Key capabilities:
  - Face detection and embedding extraction from a reference photo.
  - Face-swapping / face-consistent image synthesis (deepface / InsightFace).
  - Avatar creation (stylised portraits) from a face reference.
  - Video generation with consistent face/body across all scenes.
  - Batch processing for multiple face references simultaneously.

Integration points:
  - Uses FaceConsistencyEngine for embedding comparison.
  - Delegates image/video generation to VisualEngine or Station1Controller.
  - TODO: Wire in Veo / Grok avatar endpoints when available.
"""

from __future__ import annotations

import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger
from utils.helpers import ensure_dir, unique_filename

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------

class AvatarStyle(str, Enum):
    REALISTIC = "realistic"
    ANIMATED = "animated"
    THREE_D = "3d"
    PIXAR = "pixar"
    ANIME = "anime"
    OIL_PAINTING = "oil_painting"
    SKETCH = "sketch"


class AvatarOutputType(str, Enum):
    IMAGE = "image"
    AVATAR = "avatar"
    VIDEO = "video"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class FaceReference:
    """A user-supplied reference face image."""

    reference_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_path: Optional[Path] = None
    embedding: Optional[Any] = None  # numpy ndarray when loaded
    detected: bool = False
    confidence: float = 0.0

    def __str__(self) -> str:
        return f"FaceReference(id={self.reference_id}, detected={self.detected})"


@dataclass
class AvatarResult:
    """Result of an avatar/face-generation request."""

    success: bool
    output_type: AvatarOutputType
    output_path: Optional[Path]
    style: AvatarStyle
    reference_id: str
    provider: str = "local"
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AvatarEngine
# ---------------------------------------------------------------------------

class AvatarEngine:
    """
    Generate face-consistent images, avatars, and videos from a reference.

    Args:
        output_dir: Base directory for generated files.
        face_consistency_engine: Optional pre-built FaceConsistencyEngine.
        visual_engine: Optional VisualEngine for image/video generation.
        controller: Optional Station1Controller for API-based generation.
    """

    def __init__(
        self,
        output_dir: str = "output",
        face_consistency_engine: Optional[Any] = None,
        visual_engine: Optional[Any] = None,
        controller: Optional[Any] = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._face_engine = face_consistency_engine
        self._visual_engine = visual_engine
        self._controller = controller
        self._references: Dict[str, FaceReference] = {}

        ensure_dir(str(self._output_dir / "avatars"))
        ensure_dir(str(self._output_dir / "face_refs"))

    # ------------------------------------------------------------------ #
    # Reference Management                                                 #
    # ------------------------------------------------------------------ #

    def register_reference(self, image_path: str) -> FaceReference:
        """
        Register a face reference image.

        Steps:
          1. Copy image to a managed location.
          2. Detect face and extract embedding (if FaceConsistencyEngine available).
          3. Return a FaceReference handle for use in subsequent generation calls.

        Args:
            image_path: Path to the reference face image (JPG / PNG / WEBP).

        Returns:
            FaceReference with detection status and ID.
        """
        src = Path(image_path)
        if not src.exists():
            raise FileNotFoundError(f"Reference image not found: {image_path}")

        ref_id = str(uuid.uuid4())[:8]
        dest = self._output_dir / "face_refs" / f"{ref_id}{src.suffix}"
        shutil.copy2(src, dest)

        ref = FaceReference(reference_id=ref_id, source_path=dest)

        # Try to detect face and extract embedding
        if self._face_engine is not None:
            try:
                detections = self._face_engine.detect_faces(str(dest))
                if detections:
                    best = max(detections, key=lambda d: d.confidence)
                    ref.detected = True
                    ref.confidence = best.confidence
                    ref.embedding = best.embedding
                    log.info(
                        "Face detected in reference %s (confidence=%.2f)",
                        ref_id,
                        best.confidence,
                    )
                else:
                    log.warning("No face detected in reference image: %s", image_path)
            except Exception as exc:  # pylint: disable=broad-except
                log.error("Face detection failed for %s: %s", image_path, exc)
        else:
            log.debug("No FaceConsistencyEngine — skipping face detection.")
            ref.detected = True  # Optimistic assumption

        self._references[ref_id] = ref
        log.info("Registered face reference: %s", ref)
        return ref

    def get_reference(self, ref_id: str) -> Optional[FaceReference]:
        return self._references.get(ref_id)

    def list_references(self) -> List[FaceReference]:
        return list(self._references.values())

    # ------------------------------------------------------------------ #
    # Image Generation                                                     #
    # ------------------------------------------------------------------ #

    def generate_image(
        self,
        reference_id: str,
        prompt: str,
        style: AvatarStyle = AvatarStyle.REALISTIC,
        output_dir: Optional[str] = None,
    ) -> AvatarResult:
        """
        Generate a new image that preserves the reference face.

        Args:
            reference_id: ID of a previously registered FaceReference.
            prompt: Description of the desired scene/pose/background.
            style: Visual style for the generated image.
            output_dir: Override output directory.

        Returns:
            AvatarResult with the path to the generated image.
        """
        ref = self._references.get(reference_id)
        if ref is None:
            return AvatarResult(
                success=False,
                output_type=AvatarOutputType.IMAGE,
                output_path=None,
                style=style,
                reference_id=reference_id,
                error=f"Reference not found: {reference_id}",
            )

        out_dir = Path(output_dir or (self._output_dir / "avatars"))
        ensure_dir(str(out_dir))
        out_path = out_dir / unique_filename(f"img_{reference_id}", "png")

        full_prompt = self._build_prompt(prompt=prompt, style=style, face_ref=ref)
        log.info(
            "Generating image: ref=%s style=%s prompt='%s'",
            reference_id,
            style.value,
            full_prompt[:80],
        )

        # Try Station1Controller → VisualEngine → local stub
        try:
            if self._controller is not None:
                return self._generate_image_via_controller(
                    ref=ref,
                    prompt=full_prompt,
                    style=style,
                    out_path=out_path,
                )
            if self._visual_engine is not None:
                return self._generate_image_via_visual_engine(
                    ref=ref,
                    prompt=full_prompt,
                    style=style,
                    out_path=out_path,
                )
            # TODO: Add local diffusion model fallback (e.g. diffusers library)
            log.warning("No generation backend available — returning stub result.")
            return AvatarResult(
                success=False,
                output_type=AvatarOutputType.IMAGE,
                output_path=None,
                style=style,
                reference_id=reference_id,
                error="No image generation backend configured.",
            )
        except Exception as exc:  # pylint: disable=broad-except
            log.error("Image generation failed: %s", exc)
            return AvatarResult(
                success=False,
                output_type=AvatarOutputType.IMAGE,
                output_path=None,
                style=style,
                reference_id=reference_id,
                error=str(exc),
            )

    def _generate_image_via_controller(
        self,
        ref: FaceReference,
        prompt: str,
        style: AvatarStyle,
        out_path: Path,
    ) -> AvatarResult:
        """Delegate image generation to Station1Controller."""
        from station1_controller import TaskType  # type: ignore

        result = self._controller.run_task(
            task_type=TaskType.GENERATE_IMAGE,
            payload={
                "prompt": prompt,
                "reference_image": str(ref.source_path),
                "style": style.value,
                "output_path": str(out_path),
            },
        )
        return AvatarResult(
            success=result.success,
            output_type=AvatarOutputType.IMAGE,
            output_path=out_path if result.success else None,
            style=style,
            reference_id=ref.reference_id,
            provider=result.platform_used.value if result.platform_used else "unknown",
            error=result.error,
            metadata={"log": result.log_entries},
        )

    def _generate_image_via_visual_engine(
        self,
        ref: FaceReference,
        prompt: str,
        style: AvatarStyle,
        out_path: Path,
    ) -> AvatarResult:
        """Delegate image generation to VisualEngine."""
        generated = self._visual_engine.generate_image(
            prompt=prompt,
            output_dir=str(out_path.parent),
        )
        return AvatarResult(
            success=True,
            output_type=AvatarOutputType.IMAGE,
            output_path=generated.path,
            style=style,
            reference_id=ref.reference_id,
            provider=generated.provider,
        )

    # ------------------------------------------------------------------ #
    # Avatar Generation                                                    #
    # ------------------------------------------------------------------ #

    def generate_avatar(
        self,
        reference_id: str,
        style: AvatarStyle = AvatarStyle.REALISTIC,
        background: str = "white studio",
        output_dir: Optional[str] = None,
    ) -> AvatarResult:
        """
        Generate a styled avatar portrait from the reference face.

        TODO: Integrate dedicated avatar APIs (HeyGen, D-ID, etc.)
        """
        prompt = (
            f"Professional portrait photo of a person with the same face, "
            f"{background} background, {style.value} style, high detail, "
            f"studio lighting, sharp focus."
        )
        return self.generate_image(
            reference_id=reference_id,
            prompt=prompt,
            style=style,
            output_dir=output_dir,
        )

    # ------------------------------------------------------------------ #
    # Video Generation                                                     #
    # ------------------------------------------------------------------ #

    def generate_video(
        self,
        reference_id: str,
        scene_prompts: List[str],
        style: AvatarStyle = AvatarStyle.REALISTIC,
        aspect_ratio: str = "16:9",
        scene_seconds: float = 8.0,
        output_dir: Optional[str] = None,
    ) -> List[AvatarResult]:
        """
        Generate a multi-scene video with consistent face across all scenes.

        Each scene is generated individually and then assembled.
        Face consistency is enforced by including the reference embedding
        in each generation call.

        Args:
            reference_id: ID of the registered FaceReference.
            scene_prompts: List of per-scene description prompts.
            style: Visual style applied to all scenes.
            aspect_ratio: Output aspect ratio (e.g. "16:9", "9:16").
            scene_seconds: Duration per scene in seconds.
            output_dir: Override output directory.

        Returns:
            List of AvatarResult, one per scene.
        """
        ref = self._references.get(reference_id)
        if ref is None:
            return [
                AvatarResult(
                    success=False,
                    output_type=AvatarOutputType.VIDEO,
                    output_path=None,
                    style=style,
                    reference_id=reference_id,
                    error=f"Reference not found: {reference_id}",
                )
            ]

        out_dir = Path(output_dir or (self._output_dir / "avatars" / "videos"))
        ensure_dir(str(out_dir))

        results: List[AvatarResult] = []
        previous_video_id: Optional[str] = None  # For Grok extend-video chaining

        for idx, scene_prompt in enumerate(scene_prompts):
            log.info(
                "Generating scene %d/%d for reference %s",
                idx + 1,
                len(scene_prompts),
                reference_id,
            )
            full_prompt = self._build_prompt(
                prompt=scene_prompt, style=style, face_ref=ref
            )
            out_path = out_dir / unique_filename(
                f"scene_{idx + 1:03d}_{reference_id}", "mp4"
            )

            result = self._generate_scene_video(
                ref=ref,
                prompt=full_prompt,
                scene_idx=idx,
                style=style,
                aspect_ratio=aspect_ratio,
                scene_seconds=scene_seconds,
                out_path=out_path,
                previous_video_id=previous_video_id,
            )
            results.append(result)

            # Update chaining ID for Grok extend-video continuity
            if result.success and result.metadata.get("grok_video_id"):
                previous_video_id = result.metadata["grok_video_id"]

        return results

    def _generate_scene_video(
        self,
        ref: FaceReference,
        prompt: str,
        scene_idx: int,
        style: AvatarStyle,
        aspect_ratio: str,
        scene_seconds: float,
        out_path: Path,
        previous_video_id: Optional[str] = None,
    ) -> AvatarResult:
        """Generate a single scene video with face consistency."""
        try:
            if self._controller is not None:
                from station1_controller import TaskType  # type: ignore

                payload = {
                    "prompt": prompt,
                    "reference_image": str(ref.source_path),
                    "style": style.value,
                    "aspect_ratio": aspect_ratio,
                    "duration_seconds": scene_seconds,
                    "output_path": str(out_path),
                    "scene_index": scene_idx,
                }
                # Pass previous Grok video ID to keep scene-to-scene narrative continuity
                if previous_video_id:
                    payload["extend_from_video_id"] = previous_video_id

                ctrl_result = self._controller.run_task(
                    task_type=TaskType.IMAGE_TO_VIDEO,
                    payload=payload,
                )
                return AvatarResult(
                    success=ctrl_result.success,
                    output_type=AvatarOutputType.VIDEO,
                    output_path=out_path if ctrl_result.success else None,
                    style=style,
                    reference_id=ref.reference_id,
                    provider=ctrl_result.platform_used.value if ctrl_result.platform_used else "unknown",
                    error=ctrl_result.error,
                    metadata={
                        "log": ctrl_result.log_entries,
                        "grok_video_id": (ctrl_result.output or {}).get("extend_video_id"),
                    },
                )

            # No controller — return stub result
            log.warning("No controller available — scene video generation skipped.")
            return AvatarResult(
                success=False,
                output_type=AvatarOutputType.VIDEO,
                output_path=None,
                style=style,
                reference_id=ref.reference_id,
                error="No video generation backend configured.",
            )

        except Exception as exc:  # pylint: disable=broad-except
            log.error("Scene %d video generation failed: %s", scene_idx + 1, exc)
            return AvatarResult(
                success=False,
                output_type=AvatarOutputType.VIDEO,
                output_path=None,
                style=style,
                reference_id=ref.reference_id,
                error=str(exc),
            )

    # ------------------------------------------------------------------ #
    # Batch Processing                                                     #
    # ------------------------------------------------------------------ #

    def batch_generate(
        self,
        reference_ids: List[str],
        prompt: str,
        output_type: AvatarOutputType = AvatarOutputType.IMAGE,
        style: AvatarStyle = AvatarStyle.REALISTIC,
        **kwargs: Any,
    ) -> Dict[str, List[AvatarResult]]:
        """
        Run generation for multiple face references in batch.

        Args:
            reference_ids: List of registered reference IDs.
            prompt: Common generation prompt (applied to each reference).
            output_type: IMAGE, AVATAR, or VIDEO.
            style: Visual style for all outputs.
            **kwargs: Additional arguments forwarded to the specific generator.

        Returns:
            Dict mapping reference_id → list of AvatarResult.
        """
        results: Dict[str, List[AvatarResult]] = {}

        for ref_id in reference_ids:
            log.info("Batch processing reference: %s", ref_id)
            if output_type == AvatarOutputType.IMAGE:
                results[ref_id] = [
                    self.generate_image(
                        reference_id=ref_id, prompt=prompt, style=style, **kwargs
                    )
                ]
            elif output_type == AvatarOutputType.AVATAR:
                results[ref_id] = [
                    self.generate_avatar(
                        reference_id=ref_id, style=style, **kwargs
                    )
                ]
            elif output_type == AvatarOutputType.VIDEO:
                scene_prompts = kwargs.pop("scene_prompts", [prompt])
                results[ref_id] = self.generate_video(
                    reference_id=ref_id,
                    scene_prompts=scene_prompts,
                    style=style,
                    **kwargs,
                )

        return results

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_prompt(
        prompt: str, style: AvatarStyle, face_ref: FaceReference
    ) -> str:
        """
        Enrich a user prompt with face-consistency and style keywords.
        """
        style_tags = {
            AvatarStyle.REALISTIC: "photorealistic, hyper-detailed, 8K, professional photography",
            AvatarStyle.ANIMATED: "animated, smooth motion, vibrant colors, cartoon style",
            AvatarStyle.THREE_D: "3D render, Unreal Engine 5, cinematic lighting, ray tracing",
            AvatarStyle.PIXAR: "Pixar 3D animation, appealing character design, studio quality",
            AvatarStyle.ANIME: "anime style, cel-shaded, expressive, high detail",
            AvatarStyle.OIL_PAINTING: "oil painting, impressionist, rich textures, canvas",
            AvatarStyle.SKETCH: "pencil sketch, detailed linework, cross-hatching, monochrome",
        }
        tags = style_tags.get(style, "high quality")
        consistency = (
            "same face as the reference image, identical facial features, "
            "exact same person"
        )
        return f"{prompt}, {consistency}, {tags}"

    def face_swap(
        self,
        target_image_path: str,
        reference_id: str,
        output_dir: Optional[str] = None,
    ) -> AvatarResult:
        """
        Swap the face in target_image_path with the registered reference face.

        TODO: Integrate InsightFace / Roop / FaceSwap model.

        Args:
            target_image_path: Path to the target image whose face will be replaced.
            reference_id: ID of the source face reference.
            output_dir: Directory for the output image.

        Returns:
            AvatarResult with path to the face-swapped image.
        """
        ref = self._references.get(reference_id)
        if ref is None:
            return AvatarResult(
                success=False,
                output_type=AvatarOutputType.IMAGE,
                output_path=None,
                style=AvatarStyle.REALISTIC,
                reference_id=reference_id,
                error=f"Reference not found: {reference_id}",
            )

        out_dir = Path(output_dir or (self._output_dir / "avatars" / "swaps"))
        ensure_dir(str(out_dir))
        out_path = out_dir / unique_filename(f"swap_{reference_id}", "png")

        log.info(
            "Face swap: target=%s reference=%s output=%s",
            target_image_path,
            reference_id,
            out_path,
        )

        # TODO: Implement actual face swap using InsightFace or similar model
        return AvatarResult(
            success=False,
            output_type=AvatarOutputType.IMAGE,
            output_path=None,
            style=AvatarStyle.REALISTIC,
            reference_id=reference_id,
            error="Face swap not yet implemented — TODO: integrate InsightFace/Roop.",
        )
