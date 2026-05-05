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
from utils.helpers import ensure_dir, write_json, format_duration
from utils.validators import validate_config, validate_topic, validate_video_count

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_STYLE_ICONS = {"motion": "🌊", "cinematic": "🎥", "avatar": "🧑‍💼"}
_LANG_FLAGS = {
    "vi": "🇻🇳 Vietnamese",
    "en": "🇺🇸 English",
    "zh": "🇨🇳 Chinese",
    "ja": "🇯🇵 Japanese",
    "ko": "🇰🇷 Korean",
    "es": "🇪🇸 Spanish",
    "fr": "🇫🇷 French",
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
        st.subheader("🔑 API Keys")
        cols = st.columns(2)

        with cols[0]:
            gemini_key = st.text_input(
                "Google Gemini API Key",
                type="password",
                help="Required for script generation.",
            )
            runway_key = st.text_input(
                "Runway Gen-3 API Key",
                type="password",
                help="Required for video generation (cinematic/motion styles).",
            )
            replicate_key = st.text_input(
                "Replicate API Token",
                type="password",
                help="Required for Stable Diffusion images and Wav2Lip.",
            )

        with cols[1]:
            elevenlabs_key = st.text_input(
                "ElevenLabs API Key",
                type="password",
                help="Optional: high-quality TTS voices.",
            )
            heygen_key = st.text_input(
                "HeyGen API Key",
                type="password",
                help="Required for Avatar video style.",
            )
            budget = st.number_input(
                "Budget Limit (USD)",
                min_value=1.0,
                max_value=10000.0,
                value=300.0,
                step=10.0,
            )

        if st.form_submit_button("💾 Save Settings", type="primary"):
            # Write to .env file.
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

            if gemini_key:
                _set("GOOGLE_GEMINI_API_KEY", gemini_key)
            if runway_key:
                _set("RUNWAY_API_KEY", runway_key)
            if replicate_key:
                _set("REPLICATE_API_TOKEN", replicate_key)
            if elevenlabs_key:
                _set("ELEVENLABS_API_KEY", elevenlabs_key)
            if heygen_key:
                _set("HEYGEN_API_KEY", heygen_key)
            _set("BUDGET_LIMIT_USD", str(budget))

            env_path.write_text("\n".join(lines))
            st.success("✅ Settings saved!  Please restart the app to apply changes.")


# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------

def _render_sidebar() -> Dict[str, Any]:
    st.sidebar.title("🎬 AI Video Generator")
    st.sidebar.markdown("---")

    topic = st.sidebar.text_area(
        "📝 Video Topic",
        placeholder="E.g. 'Top 5 travel destinations in Vietnam'",
        height=100,
        help="Describe what your videos should be about.",
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
        horizontal=True,
    )

    duration = st.sidebar.select_slider(
        "⏱️ Duration (seconds)",
        options=SUPPORTED_VIDEO_DURATIONS,
        value=30,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("🔧 [Settings](#settings)")

    return {
        "topic": topic,
        "video_count": video_count,
        "styles": selected_styles,
        "variations": variations,
        "language": language,
        "aspect_ratio": aspect_ratio,
        "duration": duration,
    }


# ---------------------------------------------------------------------------
# Generation Worker (runs in background thread)
# ---------------------------------------------------------------------------

def _run_generation(params: Dict[str, Any], settings: Any) -> None:
    """Background thread: orchestrate full video generation pipeline."""
    st.session_state.generation_started = True
    st.session_state.job_status = "running"
    st.session_state.log_messages = []
    st.session_state.generated_videos = []

    try:
        _log("🚀 Starting generation pipeline …")

        brain = BrainModule(
            api_key=settings.google_gemini_api_key,
            model=settings.gemini_model,
        )
        visual = VisualEngine(
            replicate_token=settings.replicate_api_token,
            runway_api_key=settings.runway_api_key,
            heygen_api_key=settings.heygen_api_key,
            output_dir=settings.output_dir,
        )
        audio = AudioModule(
            google_credentials_json=settings.google_cloud_credentials_json,
            elevenlabs_api_key=settings.elevenlabs_api_key,
            replicate_token=settings.replicate_api_token,
            output_dir=settings.output_dir,
        )
        post = PostProduction(output_dir=settings.output_dir, fps=settings.video_fps)
        tracker = CostTracker(budget_usd=settings.budget_limit_usd)

        total_count = params["video_count"] * params["variations"]
        _log(f"📋 Generating {total_count} videos …")

        videos_done = 0
        metadata_dir = ensure_dir(f"{settings.output_dir}/metadata")

        for v_idx in range(params["video_count"]):
            for var_idx in range(params["variations"]):
                if tracker.budget_exhausted:
                    _log("🛑 Budget exhausted – stopping generation.")
                    break

                style = params["styles"][v_idx % len(params["styles"])]
                _log(
                    f"[{videos_done + 1}/{total_count}] Generating {style} video …"
                )

                # 1. Generate script.
                script = brain.generate_script(
                    topic=params["topic"],
                    style=style,
                    language=params["language"],
                    duration=params["duration"],
                    variation_index=var_idx,
                )
                tracker.record_gemini(input_tokens=600, output_tokens=400)

                # 2. Generate scenes.
                clip_paths = []
                for scene in script.scenes:
                    try:
                        if style == "avatar":
                            clip = visual.create_avatar_video(
                                script_text=scene.narration,
                                output_dir=f"{settings.output_dir}/clips",
                            )
                            tracker.record_video_generation(
                                clip.duration_seconds, provider="heygen"
                            )
                        elif style == "cinematic":
                            clip = visual.generate_cinematic_video(
                                prompt=scene.description,
                                duration_seconds=scene.duration_seconds,
                                output_dir=f"{settings.output_dir}/clips",
                            )
                            tracker.record_video_generation(
                                clip.duration_seconds, provider="runway"
                            )
                        else:  # motion
                            img = visual.generate_image(
                                prompt=scene.description,
                                output_dir=f"{settings.output_dir}/images",
                            )
                            tracker.record_image_generation(1)
                            clip = visual.generate_video_from_image(
                                image_path=img.path,
                                prompt=scene.description,
                                duration_seconds=scene.duration_seconds,
                                output_dir=f"{settings.output_dir}/clips",
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

    # Validate inputs.
    err_msgs: List[str] = []
    if not params["topic"].strip():
        err_msgs.append("Please enter a video topic in the sidebar.")
    if not settings.google_gemini_api_key:
        err_msgs.append("Google Gemini API key is not configured (Settings page).")

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

    with col_status:
        status_icon = {
            "idle": "⏸️",
            "running": "⚙️",
            "complete": "✅",
            "error": "❌",
        }.get(st.session_state.job_status, "⏸️")
        st.markdown(f"**Status:** {status_icon} {st.session_state.job_status.capitalize()}")

    # Progress bar.
    if st.session_state.generation_started:
        progress_pct = st.session_state.job_progress
        st.progress(progress_pct, text=f"Progress: {int(progress_pct * 100)}%")

    # Error message.
    if st.session_state.error_message:
        st.error(st.session_state.error_message)

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
            for msg in st.session_state.log_messages[-30:]:
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

    tab_gen, tab_results, tab_analytics = st.tabs(
        ["🚀 Generate", "📁 Results", "📊 Analytics"]
    )

    with tab_gen:
        _tab_generate(params, settings)

    with tab_results:
        _tab_results()

    with tab_analytics:
        _tab_analytics(settings)


if __name__ == "__main__":
    main()

