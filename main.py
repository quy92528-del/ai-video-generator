"""
main.py - Streamlit web interface for the AI Video Generator Suite.

Entry point:
    streamlit run main.py
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Video Generator Suite",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Local imports (after st.set_page_config)
# ---------------------------------------------------------------------------
from config import (
    PLATFORM_SCENE_SECONDS,
    SUPPORTED_ASPECT_RATIOS,
    SUPPORTED_LANGUAGES,
    SUPPORTED_VIDEO_DURATIONS,
    SUPPORTED_VIDEO_STYLES,
    get_settings,
)
from brain_module import BrainModule
from visual_engine import VisualEngine
from audio_module import AudioModule
from post_production import PostProduction
from cost_tracker import CostTracker
from parallel_processor import ParallelProcessor, BatchJob
from station1_controller import Station1Controller, Platform, TaskType
from station2_gpu_executor import Station2GPUExecutor
from avatar_engine import AvatarEngine, AvatarStyle, AvatarOutputType
from utils.helpers import ensure_dir, write_json, format_duration
from utils.validators import validate_config, validate_topic, validate_video_count

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_STYLE_ICONS = {
    "motion": "🌊",
    "cinematic": "🎥",
    "avatar": "🧑‍💼",
    "animated": "🎨",
    "3d": "🧊",
    "pixar": "🎠",
    "anime": "⛩️",
    "realistic": "📷",
    "documentary": "🎞️",
    "music_video": "🎵",
}
_LANG_FLAGS = {
    "vi": "🇻🇳 Vietnamese",
    "en": "🇺🇸 English",
    "zh": "🇨🇳 Chinese",
    "ja": "🇯🇵 Japanese",
    "ko": "🇰🇷 Korean",
    "es": "🇪🇸 Spanish",
    "fr": "🇫🇷 French",
}

_PLATFORM_ICONS = {
    "veo": "🎬 Veo",
    "grok": "⚡ Grok",
    "gemini": "✨ Gemini",
    "chatgpt": "🤖 ChatGPT",
    "both": "🔀 Veo + Grok",
}


# ---------------------------------------------------------------------------
# Session State Helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "generation_started": False,
        "generation_complete": False,
        "generated_videos": [],
        "current_job": None,
        "job_progress": 0.0,
        "job_status": "idle",
        "log_messages": [],
        "total_cost": 0.0,
        "error_message": "",
        "dark_mode": True,
        "project_name": "",
        "output_dir_override": "",
        "face_references": [],
        "avatar_results": [],
        "selected_platform": "veo",
        "platform_info": {},
        "station2_vm_status": "unknown",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    st.session_state.log_messages.append(f"[{ts}] {msg}")


# ---------------------------------------------------------------------------
# Settings page
# ---------------------------------------------------------------------------

def _settings_page() -> None:
    st.header("⚙️ Settings")
    st.markdown("Configure API keys and global preferences.")

    with st.form("settings_form"):
        st.subheader("🎬 Platform API Keys")

        st.markdown("**🎬 Veo (Google)**")
        cols_veo = st.columns(3)
        veo_key = cols_veo[0].text_input("Veo API Key (Primary)", type="password")
        veo_extra = cols_veo[1].text_input(
            "Veo API Keys (Extra, comma-separated)",
            type="password",
            help="Add multiple Veo API keys for multi-key load balancing.",
        )
        veo_account_type = cols_veo[2].text_input(
            "Veo Account Type", placeholder="e.g. Veo3_Grok"
        )
        veo_expires = st.text_input(
            "Veo Account Expiry Date", placeholder="e.g. 23/05/2027"
        )

        st.markdown("**⚡ Grok (xAI)**")
        cols_grok = st.columns(3)
        grok_key = cols_grok[0].text_input("Grok API Key (Primary)", type="password")
        grok_extra = cols_grok[1].text_input(
            "Grok API Keys (Extra, comma-separated)",
            type="password",
            help="Add multiple Grok keys for load balancing.",
        )
        grok_account_type = cols_grok[2].text_input(
            "Grok Account Type", placeholder="e.g. Grok_Pro"
        )
        grok_expires = st.text_input(
            "Grok Account Expiry Date", placeholder="e.g. 31/12/2025"
        )

        st.markdown("**✨ Gemini / Google AI**")
        cols_gemini = st.columns(2)
        gemini_key = cols_gemini[0].text_input("Google Gemini API Key", type="password")
        openai_key = cols_gemini[1].text_input(
            "🤖 OpenAI / ChatGPT API Key",
            type="password",
            help="Used for script writing with GPT-4.",
        )

        st.subheader("🔑 Other API Keys")
        cols2 = st.columns(2)
        with cols2[0]:
            runway_key = st.text_input("Runway Gen-3 API Key", type="password")
            replicate_key = st.text_input("Replicate API Token", type="password")
        with cols2[1]:
            elevenlabs_key = st.text_input("ElevenLabs API Key", type="password")
            heygen_key = st.text_input("HeyGen API Key", type="password")

        st.subheader("🖥️ Station 2 — GCP GPU VM")
        cols3 = st.columns(2)
        with cols3[0]:
            gcp_project = st.text_input(
                "GCP Project ID",
                help="Required for GPU VM management.",
            )
            gcp_zone = st.text_input(
                "GCP Zone", value="us-central1-a"
            )
            gcp_instance = st.text_input(
                "GPU Instance Name", value="ai-video-gpu-worker"
            )
        with cols3[1]:
            enable_station2 = st.checkbox(
                "Enable Station 2 GPU Fallback",
                help="When enabled, tasks fall back to a GCP GPU VM when all APIs are exhausted.",
            )

        budget = st.number_input(
            "Budget Limit (USD)",
            min_value=1.0,
            max_value=10000.0,
            value=300.0,
            step=10.0,
        )

        if st.form_submit_button("💾 Save Settings", type="primary"):
            env_path = Path(".env")
            lines: List[str] = []
            if env_path.exists():
                lines = env_path.read_text().splitlines()

            def _set(key: str, val: str) -> None:
                for i, ln in enumerate(lines):
                    if ln.startswith(f"{key}="):
                        lines[i] = f"{key}={val}"
                        return
                lines.append(f"{key}={val}")

            if veo_key:
                _set("VEO_API_KEY", veo_key)
            if veo_extra:
                _set("VEO_API_KEYS_EXTRA", veo_extra)
            if veo_account_type:
                _set("VEO_ACCOUNT_TYPE", veo_account_type)
            if veo_expires:
                _set("VEO_EXPIRES_AT", veo_expires)
            if grok_key:
                _set("GROK_API_KEY", grok_key)
            if grok_extra:
                _set("GROK_API_KEYS_EXTRA", grok_extra)
            if grok_account_type:
                _set("GROK_ACCOUNT_TYPE", grok_account_type)
            if grok_expires:
                _set("GROK_EXPIRES_AT", grok_expires)
            if gemini_key:
                _set("GOOGLE_GEMINI_API_KEY", gemini_key)
            if openai_key:
                _set("OPENAI_API_KEY", openai_key)
            if runway_key:
                _set("RUNWAY_API_KEY", runway_key)
            if replicate_key:
                _set("REPLICATE_API_TOKEN", replicate_key)
            if elevenlabs_key:
                _set("ELEVENLABS_API_KEY", elevenlabs_key)
            if heygen_key:
                _set("HEYGEN_API_KEY", heygen_key)
            if gcp_project:
                _set("GCP_PROJECT_ID", gcp_project)
            _set("GCP_ZONE", gcp_zone)
            _set("GCP_GPU_INSTANCE_NAME", gcp_instance)
            _set("ENABLE_STATION2_GPU", "true" if enable_station2 else "false")
            _set("BUDGET_LIMIT_USD", str(budget))

            env_path.write_text("\n".join(lines))
            st.success("✅ Settings saved!  Please restart the app to apply changes.")


# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------

def _render_sidebar() -> Dict[str, Any]:
    st.sidebar.title("🎬 AI Video Generator")

    # Dark / Light mode toggle
    dark_mode = st.sidebar.toggle(
        "🌙 Dark Mode",
        value=st.session_state.dark_mode,
        key="dark_mode_toggle",
    )
    st.session_state.dark_mode = dark_mode

    st.sidebar.markdown("---")

    # Optional project name (tool still runs without it)
    project_name = st.sidebar.text_input(
        "📁 Project Name (optional)",
        value=st.session_state.project_name,
        placeholder="Leave blank to use default",
        help="Name your project for organisation. Tool runs without this.",
    )
    st.session_state.project_name = project_name

    # Optional output directory override
    output_dir_override = st.sidebar.text_input(
        "💾 Save Location (optional)",
        value=st.session_state.output_dir_override,
        placeholder="Default: ./output",
        help="Choose where to save generated files.",
    )
    st.session_state.output_dir_override = output_dir_override

    st.sidebar.markdown("---")

    # --- Platform Selector ---------------------------------------------------
    st.sidebar.subheader("🚀 Platform")
    platform_options = list(_PLATFORM_ICONS.keys())
    selected_platform = st.sidebar.radio(
        "AI Video Platform",
        options=platform_options,
        format_func=lambda k: _PLATFORM_ICONS[k],
        index=platform_options.index(
            st.session_state.get("selected_platform", "veo")
        ),
        horizontal=False,
        help=(
            "Select the platform to use. "
            "Veo: 8s/clip. Grok: 6s/clip. 'Veo + Grok' uses both."
        ),
    )
    st.session_state.selected_platform = selected_platform

    st.sidebar.markdown("---")

    topic = st.sidebar.text_area(
        "📝 Video Topic / Idea",
        placeholder="E.g. 'Top 5 travel destinations in Vietnam'",
        height=100,
        help="Describe your video. The AI will expand this into a full script.",
    )

    scene_count = st.sidebar.number_input(
        "🎬 Number of Scenes",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        help="How many scenes to generate. Each scene duration depends on the platform.",
    )

    # Show calculated total duration based on platform
    if selected_platform == "both":
        # Mix: alternate between Veo and Grok
        veo_secs = PLATFORM_SCENE_SECONDS.get("veo", 8.0)
        total_dur = scene_count * veo_secs
        st.sidebar.caption(
            f"⏱️ ~{total_dur:.0f}s total ({scene_count} scenes × {veo_secs:.0f}s via Veo)"
        )
    else:
        secs = PLATFORM_SCENE_SECONDS.get(selected_platform, 8.0)
        if secs > 0:
            total_dur = scene_count * secs
            st.sidebar.caption(
                f"⏱️ ~{total_dur:.0f}s total ({scene_count} scenes × {secs:.0f}s)"
            )

    video_count = st.sidebar.slider(
        "🔢 Number of Videos",
        min_value=1,
        max_value=500,
        value=5,
        step=1,
        help="How many videos to generate.",
    )

    styles = st.sidebar.multiselect(
        "🎨 Video Styles",
        options=[f"{_STYLE_ICONS.get(s, '')} {s.capitalize()}" for s in SUPPORTED_VIDEO_STYLES],
        default=["🎥 Cinematic"],
        help="Select one or more visual styles.",
    )
    # Normalise to plain style names.
    selected_styles = [s.split()[-1].lower() for s in styles] or ["cinematic"]

    variations = st.sidebar.slider(
        "🔄 Variations per Video",
        min_value=1,
        max_value=5,
        value=1,
        help="Generate multiple variations of each script.",
    )

    language = st.sidebar.selectbox(
        "🌍 Language",
        options=list(_LANG_FLAGS.keys()),
        format_func=lambda k: _LANG_FLAGS[k],
        index=0,
    )

    aspect_ratio = st.sidebar.radio(
        "📐 Aspect Ratio",
        options=list(SUPPORTED_ASPECT_RATIOS.keys()),
        horizontal=False,
    )

    duration = st.sidebar.select_slider(
        "⏱️ Duration (seconds)",
        options=SUPPORTED_VIDEO_DURATIONS,
        value=30,
    )

    # Progress mode: Auto or Manual
    progress_mode = st.sidebar.radio(
        "⚙️ Progress Mode",
        options=["auto", "manual"],
        format_func=lambda m: "🤖 Auto" if m == "auto" else "🎛️ Manual",
        horizontal=True,
        help="Auto: AI decides scene count/duration. Manual: Use your settings above.",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("🔧 Open **Settings** page to configure API keys.")

    return {
        "topic": topic,
        "scene_count": scene_count,
        "video_count": video_count,
        "styles": selected_styles,
        "variations": variations,
        "language": language,
        "aspect_ratio": aspect_ratio,
        "duration": duration,
        "platform": selected_platform,
        "progress_mode": progress_mode,
        "project_name": project_name or "default_project",
        "output_dir": output_dir_override or "output",
    }


# ---------------------------------------------------------------------------
# Generation Worker (runs in background thread)
# ---------------------------------------------------------------------------

def _build_hybrid_controller(settings: Any) -> Station1Controller:
    """
    Build and configure the Station1Controller from current settings.

    Registers all available API keys (primary + extras) for each platform
    and wires up Station 2 GPU executor if enabled.
    """
    # --- Station 2 GPU Executor (optional) ----------------------------------
    gpu_executor: Optional[Any] = None
    if getattr(settings, "enable_station2_gpu", False):
        gpu_executor = Station2GPUExecutor(
            project_id=getattr(settings, "gcp_project_id", ""),
            zone=getattr(settings, "gcp_zone", "us-central1-a"),
            instance_name=getattr(settings, "gcp_gpu_instance_name", "ai-video-gpu-worker"),
            credentials_json=getattr(settings, "google_cloud_credentials_json", ""),
            startup_timeout=getattr(settings, "gcp_vm_startup_timeout", 300),
            task_timeout=getattr(settings, "gcp_vm_task_timeout", 3600),
            ssh_user=getattr(settings, "gcp_ssh_user", "ubuntu"),
            worker_script=getattr(settings, "gcp_worker_script", "/opt/ai-video/worker.py"),
        )

    # --- Priority from settings (e.g. "veo,grok,gemini,chatgpt") -----------
    priority_str = getattr(settings, "api_priority", "veo,grok,gemini,chatgpt")
    priority_platforms: List[Platform] = []
    for name in priority_str.split(","):
        try:
            priority_platforms.append(Platform(name.strip().lower()))
        except ValueError:
            pass

    controller = Station1Controller(
        priority=priority_platforms or None,
        gpu_executor=gpu_executor,
        status_callback=lambda s: _log(s.get("log", "")),
    )

    # --- Register Veo keys --------------------------------------------------
    veo_key = getattr(settings, "veo_api_key", "")
    if veo_key:
        controller.add_api_key(
            Platform.VEO,
            veo_key,
            label="veo_primary",
            account_type=getattr(settings, "veo_account_type", ""),
            expires_at=getattr(settings, "veo_expires_at", ""),
        )
    for extra_key in _split_keys(getattr(settings, "veo_api_keys_extra", "")):
        controller.add_api_key(
            Platform.VEO,
            extra_key,
            label="veo_extra",
            account_type=getattr(settings, "veo_account_type", ""),
            expires_at=getattr(settings, "veo_expires_at", ""),
        )

    # --- Register Grok keys -------------------------------------------------
    grok_key = getattr(settings, "grok_api_key", "")
    if grok_key:
        controller.add_api_key(
            Platform.GROK,
            grok_key,
            label="grok_primary",
            account_type=getattr(settings, "grok_account_type", ""),
            expires_at=getattr(settings, "grok_expires_at", ""),
        )
    for extra_key in _split_keys(getattr(settings, "grok_api_keys_extra", "")):
        controller.add_api_key(
            Platform.GROK,
            extra_key,
            label="grok_extra",
            account_type=getattr(settings, "grok_account_type", ""),
            expires_at=getattr(settings, "grok_expires_at", ""),
        )

    # --- Register Gemini key ------------------------------------------------
    gemini_key = getattr(settings, "google_gemini_api_key", "")
    if gemini_key:
        controller.add_api_key(Platform.GEMINI, gemini_key, label="gemini_primary")

    # --- Register ChatGPT key -----------------------------------------------
    openai_key = getattr(settings, "openai_api_key", "")
    if openai_key:
        controller.add_api_key(Platform.CHATGPT, openai_key, label="chatgpt_primary")
    for extra_key in _split_keys(getattr(settings, "openai_api_keys_extra", "")):
        controller.add_api_key(Platform.CHATGPT, extra_key, label="chatgpt_extra")

    return controller


def _split_keys(raw: str) -> List[str]:
    """Split a comma-separated string of API keys into a list, ignoring empty."""
    return [k.strip() for k in raw.split(",") if k.strip()]


def _platform_priority_from_selection(selected: str) -> Optional[List[Platform]]:
    """Map a UI platform selection to a priority list."""
    mapping: Dict[str, List[Platform]] = {
        "veo": [Platform.VEO, Platform.GROK, Platform.GEMINI, Platform.CHATGPT],
        "grok": [Platform.GROK, Platform.VEO, Platform.GEMINI, Platform.CHATGPT],
        "gemini": [Platform.GEMINI, Platform.VEO, Platform.GROK, Platform.CHATGPT],
        "chatgpt": [Platform.CHATGPT, Platform.GEMINI, Platform.GROK, Platform.VEO],
        "both": [Platform.VEO, Platform.GROK, Platform.GEMINI, Platform.CHATGPT],
    }
    return mapping.get(selected)


# ---------------------------------------------------------------------------
# Generation Worker (runs in background thread)
# ---------------------------------------------------------------------------

def _run_generation(params: Dict[str, Any], settings: Any) -> None:
    """Background thread: orchestrate full video generation pipeline via Hybrid controller."""
    st.session_state.generation_started = True
    st.session_state.job_status = "running"
    st.session_state.log_messages = []
    st.session_state.generated_videos = []

    try:
        _log("🚀 Starting Hybrid generation pipeline …")

        # Build controller with platform priority matching UI selection
        priority = _platform_priority_from_selection(params.get("platform", "veo"))
        controller = _build_hybrid_controller(settings)
        if priority:
            controller._priority = priority  # apply UI selection

        output_dir = params.get("output_dir") or settings.output_dir

        brain = BrainModule(
            api_key=settings.google_gemini_api_key,
            model=settings.gemini_model,
        )
        visual = VisualEngine(
            replicate_token=settings.replicate_api_token,
            runway_api_key=settings.runway_api_key,
            heygen_api_key=settings.heygen_api_key,
            output_dir=output_dir,
        )
        audio = AudioModule(
            google_credentials_json=settings.google_cloud_credentials_json,
            elevenlabs_api_key=settings.elevenlabs_api_key,
            replicate_token=settings.replicate_api_token,
            output_dir=output_dir,
        )
        post = PostProduction(output_dir=output_dir, fps=settings.video_fps)
        tracker = CostTracker(budget_usd=settings.budget_limit_usd)

        total_count = params["video_count"] * params["variations"]
        _log(f"📋 Generating {total_count} videos via platform: {params.get('platform', 'veo')} …")

        videos_done = 0
        metadata_dir = ensure_dir(f"{output_dir}/metadata")

        for v_idx in range(params["video_count"]):
            for var_idx in range(params["variations"]):
                if tracker.budget_exhausted:
                    _log("🛑 Budget exhausted – stopping generation.")
                    break

                style = params["styles"][v_idx % len(params["styles"])]
                _log(
                    f"[{videos_done + 1}/{total_count}] Generating {style} video …"
                )

                # 1. Generate script via controller (Gemini/ChatGPT/Grok).
                script_result = controller.run_task(
                    task_type=TaskType.WRITE_SCRIPT,
                    payload={
                        "topic": params["topic"],
                        "style": style,
                        "language": params["language"],
                        "duration": params["duration"],
                        "scene_count": params.get("scene_count", 10),
                        "variation_index": var_idx,
                    },
                )
                # Fall back to BrainModule if controller script is a stub
                script = brain.generate_script(
                    topic=params["topic"],
                    style=style,
                    language=params["language"],
                    duration=params["duration"],
                    variation_index=var_idx,
                )
                tracker.record_gemini(input_tokens=600, output_tokens=400)
                _log(
                    f"  ✍️ Script via {script_result.platform_used or 'gemini'}: "
                    f"{len(script.scenes)} scenes"
                )

                # 2. Generate scenes via controller (Veo / Grok / GPU fallback).
                clip_paths = []
                previous_video_id: Optional[str] = None

                for scene in script.scenes:
                    payload: Dict[str, Any] = {
                        "prompt": scene.description,
                        "narration": scene.narration,
                        "duration_seconds": scene.duration_seconds,
                        "aspect_ratio": params["aspect_ratio"],
                        "style": style,
                        "scene_index": scene.index,
                        "scene_count": len(script.scenes),
                    }
                    if previous_video_id:
                        payload["extend_from_video_id"] = previous_video_id

                    ctrl_result = controller.run_task(
                        task_type=TaskType.TEXT_TO_VIDEO
                        if not hasattr(scene, "image_path")
                        else TaskType.IMAGE_TO_VIDEO,
                        payload=payload,
                    )

                    # Update chaining ID for Grok extend-video continuity
                    if ctrl_result.output and isinstance(ctrl_result.output, dict):
                        previous_video_id = ctrl_result.output.get("extend_video_id")

                    # Visual engine fallback for scene clip (local processing)
                    try:
                        if style == "avatar":
                            clip = visual.create_avatar_video(
                                script_text=scene.narration,
                                output_dir=f"{output_dir}/clips",
                            )
                            tracker.record_video_generation(
                                clip.duration_seconds, provider="heygen"
                            )
                        elif style == "cinematic":
                            clip = visual.generate_cinematic_video(
                                prompt=scene.description,
                                duration_seconds=scene.duration_seconds,
                                output_dir=f"{output_dir}/clips",
                            )
                            tracker.record_video_generation(
                                clip.duration_seconds, provider="runway"
                            )
                        else:  # motion / default
                            img = visual.generate_image(
                                prompt=scene.description,
                                output_dir=f"{output_dir}/images",
                            )
                            tracker.record_image_generation(1)
                            clip = visual.generate_video_from_image(
                                image_path=img.path,
                                prompt=scene.description,
                                duration_seconds=scene.duration_seconds,
                                output_dir=f"{output_dir}/clips",
                            )
                            tracker.record_video_generation(
                                clip.duration_seconds, provider="runway"
                            )
                        clip_paths.append(clip.path)
                    except Exception as exc:  # pylint: disable=broad-except
                        _log(f"  ⚠️ Scene {scene.index} failed: {exc}")

                # 3. Generate audio.
                all_narration = " ".join(s.narration for s in script.scenes)
                voice_track = audio.synthesize_speech(
                    text=all_narration,
                    language=params["language"],
                )
                tracker.record_tts(len(all_narration), provider="google_tts")

                mixed = audio.mix_audio(voice_track=voice_track)

                # 4. Assemble video.
                result = post.assemble_video(
                    clip_paths=clip_paths,
                    audio_path=mixed.path,
                    aspect_ratio=params["aspect_ratio"],
                    title=script.title,
                )
                tracker.increment_videos(1)

                # 5. Save metadata.
                meta = {
                    **script.to_dict(),
                    "video_path": str(result.path),
                    "thumbnail_path": str(result.thumbnail_path) if result.thumbnail_path else None,
                    "duration_seconds": result.duration_seconds,
                    "aspect_ratio": params["aspect_ratio"],
                    "cost_usd": tracker.total_cost,
                    "platform": params.get("platform", "veo"),
                    "scene_count": len(script.scenes),
                    "project_name": params.get("project_name", "default_project"),
                }
                meta_path = metadata_dir / f"video_{v_idx:04d}_var{var_idx}.json"
                write_json(meta, meta_path)

                st.session_state.generated_videos.append(
                    {
                        "video_path": str(result.path),
                        "thumbnail_path": str(result.thumbnail_path) if result.thumbnail_path else "",
                        "title": script.title,
                        "style": style,
                        "metadata": meta,
                    }
                )

                videos_done += 1
                st.session_state.job_progress = videos_done / total_count
                st.session_state.total_cost = tracker.total_cost

            else:
                continue
            break  # budget exhausted inner loop – propagate to outer

        # Save cost report.
        tracker.save_report()

        _log(
            f"✅ Generation complete! {videos_done} videos | "
            f"${tracker.total_cost:.2f} spent."
        )
        st.session_state.job_status = "complete"
        st.session_state.generation_complete = True

    except Exception as exc:  # pylint: disable=broad-except
        _log(f"❌ Fatal error: {exc}")
        st.session_state.job_status = "error"
        st.session_state.error_message = str(exc)


# ---------------------------------------------------------------------------
# Tab: Generate
# ---------------------------------------------------------------------------

def _tab_generate(params: Dict[str, Any], settings: Any) -> None:
    st.header("🚀 Generate Videos")

    # --- Platform & Account Info -------------------------------------------
    with st.expander("🔑 Platform & Account Info", expanded=False):
        platform = params.get("platform", "veo")
        if platform in ("veo", "both"):
            veo_acct = getattr(settings, "veo_account_type", "") or "N/A"
            veo_exp = getattr(settings, "veo_expires_at", "") or "N/A"
            st.markdown(
                f"🎬 **Veo** — Account type: **{veo_acct}** | Expiry: **{veo_exp}** "
                f"| Clip duration: **8s/scene**"
            )
        if platform in ("grok", "both"):
            grok_acct = getattr(settings, "grok_account_type", "") or "N/A"
            grok_exp = getattr(settings, "grok_expires_at", "") or "N/A"
            st.markdown(
                f"⚡ **Grok** — Account type: **{grok_acct}** | Expiry: **{grok_exp}** "
                f"| Clip duration: **6s/scene**"
            )
        scene_count = params.get("scene_count", 10)
        secs = PLATFORM_SCENE_SECONDS.get(
            "veo" if platform in ("veo", "both") else platform, 8.0
        )
        total_secs = scene_count * secs
        st.info(
            f"⏱️ Total ~{total_secs:.0f}s ({scene_count} scenes × {secs:.0f}s)"
        )

    # Validate inputs.
    err_msgs: List[str] = []
    if not params["topic"].strip():
        err_msgs.append("Please enter a video topic in the sidebar.")
    if (
        not getattr(settings, "veo_api_key", "")
        and not getattr(settings, "grok_api_key", "")
        and not settings.google_gemini_api_key
    ):
        err_msgs.append(
            "No API keys configured. Add at least one key on the Settings page."
        )

    if err_msgs:
        for msg in err_msgs:
            st.warning(msg)

    col_btn, col_status = st.columns([2, 5])
    with col_btn:
        can_start = not st.session_state.generation_started or st.session_state.generation_complete
        if st.button(
            "▶️ Start Generation",
            type="primary",
            disabled=not can_start or bool(err_msgs),
            use_container_width=True,
        ):
            st.session_state.generation_started = True
            st.session_state.generation_complete = False
            st.session_state.job_progress = 0.0
            thread = threading.Thread(
                target=_run_generation,
                args=(params, settings),
                daemon=True,
            )
            thread.start()
            st.rerun()

        if st.session_state.generation_started and not st.session_state.generation_complete:
            if st.button("⏹️ Stop", use_container_width=True):
                st.session_state.job_status = "idle"
                st.session_state.generation_started = False
                st.rerun()

    with col_status:
        status_icon = {
            "idle": "⏸️",
            "running": "⚙️",
            "complete": "✅",
            "error": "❌",
        }.get(st.session_state.job_status, "⏸️")
        st.markdown(f"**Status:** {status_icon} {st.session_state.job_status.capitalize()}")
        if params.get("project_name"):
            st.caption(f"📁 Project: {params['project_name']}")

    # Progress bar.
    if st.session_state.generation_started:
        progress_pct = st.session_state.job_progress
        st.progress(progress_pct, text=f"Progress: {int(progress_pct * 100)}%")

    # Error message.
    if st.session_state.error_message:
        st.error(st.session_state.error_message)
        if st.button("🔄 Retry Last Failed Video"):
            st.session_state.error_message = ""
            thread = threading.Thread(
                target=_run_generation,
                args=(params, settings),
                daemon=True,
            )
            thread.start()
            st.rerun()

    # Live cost.
    if st.session_state.total_cost > 0:
        st.metric(
            "💰 Estimated Cost",
            f"${st.session_state.total_cost:.4f}",
            delta=f"Budget: ${settings.budget_limit_usd:.0f}",
        )

    # Log output.
    if st.session_state.log_messages:
        with st.expander("📋 Generation Log", expanded=True):
            for msg in st.session_state.log_messages[-50:]:
                st.text(msg)

    # Cost estimate preview.
    with st.expander("💡 Cost Estimate Preview"):
        tracker = CostTracker(budget_usd=settings.budget_limit_usd)
        for style in params["styles"][:3]:
            est = tracker.estimate_cost_per_video(style, params["duration"])
            st.markdown(f"**{style.capitalize()} style** – est. ${est['total']:.3f}/video")
            st.json(est)

    # Auto-refresh while running.
    if st.session_state.job_status == "running":
        time.sleep(2)
        st.rerun()


# ---------------------------------------------------------------------------
# Tab: Avatar / Face Generator
# ---------------------------------------------------------------------------

def _tab_avatar(settings: Any) -> None:
    st.header("🧑‍💼 Avatar & Face Video Generator")
    st.markdown(
        "Upload a reference face photo, then generate AI images, avatars, "
        "or face-consistent videos."
    )

    output_dir = st.session_state.get("output_dir_override") or getattr(settings, "output_dir", "output")

    col_upload, col_options = st.columns([1, 2])

    with col_upload:
        st.subheader("📷 Reference Face")
        uploaded_file = st.file_uploader(
            "Upload face reference image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a clear front-facing photo of the person.",
        )
        if uploaded_file is not None:
            # Save uploaded file temporarily
            ref_dir = Path(output_dir) / "face_refs"
            ref_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = ref_dir / f"upload_{uploaded_file.name}"
            tmp_path.write_bytes(uploaded_file.getvalue())

            st.image(str(tmp_path), caption="Reference face", width=200)

            if st.button("✅ Register Face Reference", type="primary"):
                avatar_engine = AvatarEngine(output_dir=output_dir)
                ref = avatar_engine.register_reference(str(tmp_path))
                refs = st.session_state.face_references
                refs.append(
                    {"id": ref.reference_id, "path": str(tmp_path), "detected": ref.detected}
                )
                st.session_state.face_references = refs
                if ref.detected:
                    st.success(f"✅ Face registered! ID: {ref.reference_id}")
                else:
                    st.warning(
                        f"⚠️ Face registered (ID: {ref.reference_id}) but detection uncertain — "
                        "results may vary."
                    )

        # Show registered references
        if st.session_state.face_references:
            st.markdown("**Registered faces:**")
            for r in st.session_state.face_references:
                st.caption(
                    f"ID: {r['id']} | Detected: {'✅' if r['detected'] else '⚠️'}"
                )

    with col_options:
        st.subheader("🎨 Generation Options")

        ref_ids = [r["id"] for r in st.session_state.face_references]
        if not ref_ids:
            st.info("Upload and register a face reference first.")
            return

        selected_ref = st.selectbox("Select face reference", ref_ids)

        output_type = st.radio(
            "Output type",
            options=[AvatarOutputType.IMAGE, AvatarOutputType.AVATAR, AvatarOutputType.VIDEO],
            format_func=lambda t: {
                AvatarOutputType.IMAGE: "🖼️ Image",
                AvatarOutputType.AVATAR: "🧑‍💼 Avatar",
                AvatarOutputType.VIDEO: "🎬 Video",
            }.get(t, str(t)),
            horizontal=True,
        )

        style = st.selectbox(
            "Style",
            options=list(AvatarStyle),
            format_func=lambda s: s.value.replace("_", " ").title(),
        )

        prompt = st.text_area(
            "Scene / Prompt",
            placeholder="Describe the scene, pose, background, action…",
            height=80,
        )

        if output_type == AvatarOutputType.VIDEO:
            scene_prompts_raw = st.text_area(
                "Scene prompts (one per line)",
                placeholder="Scene 1 description\nScene 2 description\n…",
                height=120,
                help="Each line becomes one scene clip.",
            )
            aspect_ratio = st.selectbox(
                "Aspect Ratio",
                options=list(SUPPORTED_ASPECT_RATIOS.keys()),
                index=1,
            )
            scene_secs = st.number_input(
                "Seconds per scene",
                min_value=1.0,
                max_value=60.0,
                value=8.0,
                step=1.0,
            )

        if st.button("🚀 Generate", type="primary"):
            avatar_engine = AvatarEngine(output_dir=output_dir)
            # Re-register the reference so the engine knows about it
            ref_info = next(
                (r for r in st.session_state.face_references if r["id"] == selected_ref),
                None,
            )
            if ref_info:
                avatar_engine.register_reference(ref_info["path"])

            with st.spinner("Generating…"):
                if output_type == AvatarOutputType.IMAGE:
                    result = avatar_engine.generate_image(
                        reference_id=selected_ref, prompt=prompt, style=style
                    )
                    if result.success and result.output_path and result.output_path.exists():
                        st.image(str(result.output_path), caption="Generated Image")
                    else:
                        st.error(f"Generation failed: {result.error}")

                elif output_type == AvatarOutputType.AVATAR:
                    result = avatar_engine.generate_avatar(
                        reference_id=selected_ref, style=style
                    )
                    if result.success and result.output_path and result.output_path.exists():
                        st.image(str(result.output_path), caption="Generated Avatar")
                    else:
                        st.error(f"Generation failed: {result.error}")

                elif output_type == AvatarOutputType.VIDEO:
                    scenes = [
                        s.strip()
                        for s in scene_prompts_raw.splitlines()
                        if s.strip()
                    ] or [prompt]
                    results = avatar_engine.generate_video(
                        reference_id=selected_ref,
                        scene_prompts=scenes,
                        style=style,
                        aspect_ratio=aspect_ratio,
                        scene_seconds=scene_secs,
                    )
                    for i, r in enumerate(results):
                        if r.success and r.output_path and r.output_path.exists():
                            st.video(str(r.output_path))
                        else:
                            st.warning(
                                f"Scene {i + 1} failed: {r.error}. "
                                "Re-run generation to retry failed scenes."
                            )


# ---------------------------------------------------------------------------
# Tab: Results
# ---------------------------------------------------------------------------

def _tab_results() -> None:
    st.header("📁 Results")

    videos = st.session_state.generated_videos
    if not videos:
        st.info("No videos generated yet.  Go to the **Generate** tab to start.")
        return

    st.success(f"✅ {len(videos)} video(s) ready for download.")

    for i, vid in enumerate(videos):
        with st.expander(f"🎬 Video {i + 1}: {vid['title'][:60]}", expanded=i < 3):
            cols = st.columns([2, 3])
            with cols[0]:
                if vid.get("thumbnail_path") and Path(vid["thumbnail_path"]).exists():
                    st.image(vid["thumbnail_path"], caption="Thumbnail")
                else:
                    st.markdown("*(No thumbnail)*")

            with cols[1]:
                st.markdown(f"**Style:** {_STYLE_ICONS.get(vid['style'], '')} {vid['style']}")
                st.markdown(f"**Title:** {vid['title']}")

                video_path = Path(vid["video_path"])
                if video_path.exists() and video_path.stat().st_size > 0:
                    with open(video_path, "rb") as f:
                        st.download_button(
                            f"⬇️ Download Video",
                            data=f.read(),
                            file_name=video_path.name,
                            mime="video/mp4",
                            key=f"dl_video_{i}",
                        )
                else:
                    st.caption("*(Video file not available)*")

                meta_json = json.dumps(vid["metadata"], indent=2, ensure_ascii=False)
                st.download_button(
                    "⬇️ Download Metadata",
                    data=meta_json.encode(),
                    file_name=f"metadata_{i}.json",
                    mime="application/json",
                    key=f"dl_meta_{i}",
                )


# ---------------------------------------------------------------------------
# Tab: Analytics
# ---------------------------------------------------------------------------

def _tab_analytics(settings: Any) -> None:
    st.header("📊 Analytics & Cost Tracking")

    tracker = CostTracker(budget_usd=settings.budget_limit_usd)
    summary = tracker.get_summary()
    videos = st.session_state.generated_videos
    total_cost = st.session_state.total_cost

    # KPI cards.
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🎬 Videos Generated", len(videos))
    k2.metric("💰 Total Cost", f"${total_cost:.3f}")
    k3.metric(
        "📊 Budget Used",
        f"{total_cost / settings.budget_limit_usd * 100:.1f}%",
        delta=f"${settings.budget_limit_usd - total_cost:.2f} remaining",
    )
    k4.metric(
        "💵 Cost per Video",
        f"${total_cost / len(videos):.3f}" if videos else "$0.000",
    )

    st.markdown("---")

    # Cost estimate table.
    st.subheader("💡 Estimated Cost by Style & Duration")
    _tracker = CostTracker()
    rows = []
    for style in SUPPORTED_VIDEO_STYLES:
        for dur in [15, 30, 60, 120]:
            est = _tracker.estimate_cost_per_video(style, dur)
            rows.append(
                {
                    "Style": style,
                    "Duration (s)": dur,
                    "Script": f"${est.get('script_gemini', 0):.4f}",
                    "Images/Video": f"${est.get('images_sd', est.get('video_heygen', 0)):.4f}",
                    "Clips": f"${est.get('video_runway', 0):.4f}",
                    "TTS": f"${est.get('tts_google', 0):.4f}",
                    "Total": f"${est['total']:.4f}",
                }
            )

    try:
        import pandas as pd  # type: ignore

        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    except ImportError:
        st.json(rows)

    # Budget slider preview.
    st.subheader("🎯 Videos per Budget")
    budget_input = st.slider(
        "Budget (USD)",
        min_value=10,
        max_value=300,
        value=int(settings.budget_limit_usd),
        step=10,
    )
    for style in SUPPORTED_VIDEO_STYLES:
        est = CostTracker().estimate_cost_per_video(style, 30)
        if est["total"] > 0:
            count = int(budget_input / est["total"])
            st.markdown(
                f"**{_STYLE_ICONS.get(style, '')} {style.capitalize()}** "
                f"(30s): ~**{count}** videos for ${budget_input}"
            )


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main() -> None:
    _init_state()

    try:
        settings = get_settings()
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Failed to load settings: {exc}")
        return

    # Navigation.
    page = st.sidebar.radio(
        "Navigation",
        ["🎬 Generator", "⚙️ Settings"],
        label_visibility="collapsed",
    )

    if page == "⚙️ Settings":
        _settings_page()
        return

    # Generator layout.
    params = _render_sidebar()

    tab_gen, tab_avatar, tab_results, tab_analytics = st.tabs(
        ["🚀 Generate", "🧑‍💼 Avatar", "📁 Results", "📊 Analytics"]
    )

    with tab_gen:
        _tab_generate(params, settings)

    with tab_avatar:
        _tab_avatar(settings)

    with tab_results:
        _tab_results()

    with tab_analytics:
        _tab_analytics(settings)


if __name__ == "__main__":
    main()
