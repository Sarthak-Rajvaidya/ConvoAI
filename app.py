"""
app.py
===============================================================================
AI Meeting Intelligence Assistant — Streamlit entry point.

Pipeline:

    YouTube / Audio / Video
            |
            v
    Fast audio acquisition (services/audio)
            |
            v
    Groq Whisper Large V3 Turbo transcription (services/transcription)
            |
            v
    ONE structured Groq analysis call (services/analysis)
            |
            v
    Results shown immediately  <-- RAG indexing runs in the background
            |
            v
    Streaming, source-cited RAG chat (services/rag)

Run:
    streamlit run app.py
===============================================================================
"""

from __future__ import annotations

import streamlit as st

from config.settings import get_settings
from ui.analyze import render_analyze_page
from ui.chat import render_chat_page
from ui.results import render_results_page
from ui.styles import inject_css
from ui.transcript import render_transcript_page

st.set_page_config(
    page_title="AI Meeting Intelligence Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()


# =============================================================================
# SESSION STATE
# =============================================================================

def init_state() -> None:
    defaults = {
        "result": None,
        "meeting_id": None,
        "chat_messages": [],
        "active_page": "Analyze",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()
settings = get_settings()


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### 🧠 AI Meeting Intelligence")
    st.caption("Meetings, lectures & videos → searchable, actionable intelligence")
    st.divider()

    pages = ["Analyze", "Results", "Transcript", "Chat with Meeting"]
    if st.session_state.active_page not in pages:
        st.session_state.active_page = "Analyze"

    st.session_state.active_page = st.radio(
        "Navigation", pages, index=pages.index(st.session_state.active_page),
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("#### System Status")

    status_rows = [
        ("Groq (transcription + analysis)", settings.groq_configured),
        ("Qdrant (semantic search)", settings.qdrant_configured),
    ]
    for label, ok in status_rows:
        st.markdown(f"{'🟢' if ok else '🔴'} {label}")

    if not settings.groq_configured:
        st.caption("Add GROQ_API_KEY to your .env file to enable transcription, analysis and chat.")
    if not settings.qdrant_configured:
        st.caption("Qdrant not configured — chat will use transcript-based search instead of semantic search.")

    if settings.demo_mode:
        st.markdown("🧪 **Demo Mode is ON**")

    st.divider()
    st.caption("v3.0 · Groq Whisper · Qdrant · Structured RAG")


# =============================================================================
# ROUTER
# =============================================================================

page = st.session_state.active_page

if page == "Analyze":
    render_analyze_page()
elif page == "Results":
    render_results_page()
elif page == "Transcript":
    render_transcript_page()
elif page == "Chat with Meeting":
    render_chat_page()
