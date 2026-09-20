"""
ui/analyze.py
===============================================================================
Landing / Analyze page: hero, feature grid, source input (YouTube or file
upload), language/output controls, and the processing experience.
===============================================================================
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from config.settings import get_settings
from services.pipeline import run_pipeline
from ui.components import render_feature_grid, render_timeline
from utils.demo_data import build_demo_result
from utils.errors import AppError
from utils.logging import get_logger
from utils.validation import is_youtube_url

logger = get_logger(__name__)

SUPPORTED_FORMATS = ["mp3", "wav", "m4a", "mp4", "webm", "mov", "mkv", "aac", "ogg", "flac"]

FEATURES = [
    ("⚡", "Fast Transcription", "Groq-hosted Whisper Large V3 Turbo — cloud speech recognition, no local model download."),
    ("🧠", "Meeting Intelligence", "One structured AI pass extracts summary, decisions, risks and action items."),
    ("🔎", "Semantic Search", "Qdrant-backed vector search over the transcript, indexed in the background."),
    ("💬", "RAG Chat", "Ask questions and get streaming, source-cited answers grounded in the transcript."),
    ("📋", "Action Items", "Task, owner, deadline and priority extracted automatically."),
    ("📊", "Decision Tracking", "Decisions, open questions and risks surfaced as structured, reviewable lists."),
]


def render_hero() -> None:
    st.markdown('<div class="app-title">AI Meeting Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Turn meetings into decisions, tasks and searchable knowledge.</div>',
        unsafe_allow_html=True,
    )


def render_analyze_page() -> None:
    settings = get_settings()
    render_hero()

    if settings.demo_mode:
        st.info(
            "**Demo Mode is on.** Click below to load a sample meeting — no API calls, "
            "no credits used.",
            icon="🧪",
        )
        if st.button("Load demo meeting", type="primary"):
            _finish_with_result(build_demo_result())
        st.divider()

    tab_yt, tab_file = st.tabs(["📺 YouTube URL", "📁 Upload File"])

    source: str | None = None
    is_youtube = False
    uploaded_file = None

    with tab_yt:
        yt_url = st.text_input(
            "Paste a YouTube URL", placeholder="https://youtube.com/watch?v=...",
            key="yt_url_input",
        )
        if yt_url and yt_url.strip():
            source = yt_url.strip()
            is_youtube = True

    with tab_file:
        uploaded_file = st.file_uploader(
            "Upload audio or video", type=SUPPORTED_FORMATS, key="file_uploader"
        )
        if uploaded_file is not None:
            size_mb = uploaded_file.size / 1_000_000
            st.caption(f"Ready: {uploaded_file.name} ({size_mb:.1f} MB)")
            source = uploaded_file.name
            is_youtube = False

    with st.expander("Transcription options"):
        col1, col2 = st.columns(2)
        with col1:
            lang_choice = st.selectbox(
                "Transcription language", ["Auto Detect", "English", "Hindi", "Spanish", "French", "German"],
                key="lang_choice",
            )
        with col2:
            output_choice = st.selectbox(
                "Output", ["Original Language", "English Translation"], key="output_choice"
            )

    lang_map = {"Auto Detect": None, "English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr", "German": "de"}
    language_hint = lang_map.get(lang_choice)
    translate = output_choice == "English Translation"

    analyze_clicked = st.button("🚀 Analyze Meeting", type="primary", use_container_width=False)

    if analyze_clicked:
        if not source:
            st.warning("Paste a YouTube URL or upload a file first.")
        elif is_youtube and not is_youtube_url(source):
            st.warning("That doesn't look like a supported YouTube URL.")
        elif not settings.groq_configured:
            st.error(
                "GROQ_API_KEY isn't configured. Add it to your `.env` file to enable "
                "transcription and analysis."
            )
        else:
            _run_with_progress(
                source=source, is_youtube=is_youtube, uploaded_file=uploaded_file,
                settings=settings, language_hint=language_hint, translate=translate,
            )
            return

    st.divider()
    st.markdown('<div class="section-label">What you get</div>', unsafe_allow_html=True)
    render_feature_grid(FEATURES)


def _run_with_progress(*, source, is_youtube, uploaded_file, settings, language_hint, translate):
    uploaded_tmp_path = None
    uploaded_size = None
    uploaded_filename = None

    if uploaded_file is not None:
        tmp_dir = Path("tmp_uploads")
        tmp_dir.mkdir(exist_ok=True)
        uploaded_tmp_path = tmp_dir / f"upload_{int(time.time())}_{uploaded_file.name}"
        with open(uploaded_tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        uploaded_size = uploaded_file.size
        uploaded_filename = uploaded_file.name

    stage_order = ["audio", "transcription", "analysis", "indexing"]
    completed: set[str] = set()
    started_at = time.time()

    with st.status("Meeting Processing", expanded=True) as status:
        timeline_ph = st.empty()
        elapsed_ph = st.empty()

        def progress_callback(stage: str):
            idx = stage_order.index(stage) if stage in stage_order else 0
            completed.update(stage_order[:idx])
            with timeline_ph.container():
                render_timeline(stage, completed)
            elapsed_ph.caption(f"Elapsed: {time.time() - started_at:.1f}s")

        try:
            result = run_pipeline(
                source, is_youtube=is_youtube,
                uploaded_tmp_path=uploaded_tmp_path, uploaded_filename=uploaded_filename,
                uploaded_size=uploaded_size, settings=settings,
                language_hint=language_hint, translate_to_english=translate,
                progress_callback=progress_callback,
            )
            completed.update({"audio", "transcription", "analysis"})
            with timeline_ph.container():
                render_timeline("indexing", completed)
            status.update(
                label=f"Meeting Intelligence Ready · {time.time() - started_at:.1f}s",
                state="complete", expanded=False,
            )
        except AppError as exc:
            status.update(label="Processing failed", state="error", expanded=True)
            st.error(f"**{exc.user_message}**")
            if exc.detail:
                with st.expander("Technical details"):
                    st.code(exc.detail)
            if st.button("Try again"):
                st.rerun()
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected pipeline failure")
            status.update(label="Processing failed", state="error", expanded=True)
            st.error("Something unexpected went wrong while processing this meeting.")
            with st.expander("Technical details"):
                st.code(str(exc))
            if st.button("Try again"):
                st.rerun()
            return
        finally:
            if uploaded_tmp_path is not None:
                try:
                    uploaded_tmp_path.unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass

    _finish_with_result(result)


def _finish_with_result(result) -> None:
    st.session_state.result = result
    st.session_state.meeting_id = result.meeting_id
    st.session_state.chat_messages = []
    st.session_state.active_page = "Results"
    st.success("Meeting intelligence is ready.")
    st.rerun()
