"""
ui/chat.py
===============================================================================
The dedicated AI chat page: streaming answers, source citations, bounded
conversation memory, suggested questions, and a live "knowledge base"
status indicator that polls the background indexing job with
st.fragment(run_every=...).
===============================================================================
"""

from __future__ import annotations

import streamlit as st

from config.settings import get_settings
from models.meeting import MeetingResult
from models.rag import ChatTurn
from services.rag.chat import MeetingChat
from services.rag.memory_store import rag_status_tracker
from ui.components import empty_state, render_badges, render_sources
from utils.errors import AppError
from utils.logging import get_logger

logger = get_logger(__name__)

SUGGESTED_QUESTIONS = [
    "What were the key decisions?",
    "Who owns the action items?",
    "What deadlines were mentioned?",
    "What risks were discussed?",
]


@st.fragment(run_every=2)
def _render_status(meeting_id: str) -> None:
    status = rag_status_tracker.get(meeting_id)
    label, kind = {
        "pending": ("Knowledge Base: Preparing…", "neutral"),
        "indexing": ("Knowledge Base: Indexing…", "warning"),
        "ready": ("RAG Ready", "success"),
        "unavailable": ("Using transcript-based search", "neutral"),
        "error": ("Indexing failed — using transcript search", "danger"),
    }.get(status.state, ("Knowledge Base: Preparing…", "neutral"))
    if status.state == "ready":
        label = f"RAG Ready · {status.chunk_count} chunks indexed"
    render_badges((label, kind))


def render_chat_page() -> None:
    result: MeetingResult | None = st.session_state.get("result")
    if result is None:
        empty_state("💬", "No meeting loaded", "Analyze a meeting first, then come back to chat with it.")
        return

    st.markdown('<div class="app-title">Meeting AI</div>', unsafe_allow_html=True)
    st.caption(result.analysis.title)
    _render_status(result.meeting_id)
    st.divider()

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    clear_col, _ = st.columns([1, 4])
    with clear_col:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()

    pending_question = None
    if not st.session_state.chat_messages:
        st.caption("Try asking:")
        cols = st.columns(len(SUGGESTED_QUESTIONS))
        for col, q in zip(cols, SUGGESTED_QUESTIONS):
            with col:
                if st.button(q, key=f"suggested_{q}", use_container_width=True):
                    pending_question = q

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []), message.get("fallback_note", ""))

    question = st.chat_input("Ask about this meeting...") or pending_question
    if not question or not question.strip():
        return

    question = question.strip()
    st.session_state.chat_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        settings = get_settings()

        try:
            chat = MeetingChat(settings)
            history = [
                ChatTurn(role=m["role"], content=m["content"])
                for m in st.session_state.chat_messages[:-1]
                if m["role"] in ("user", "assistant")
            ]
            stream, sources, used_fallback, fallback_reason = chat.stream_answer(
                question, result.meeting_id, history
            )

            full_answer = ""
            for token in stream:
                full_answer += token
                placeholder.markdown(full_answer + "▌")
            placeholder.markdown(full_answer)

            render_sources(sources, fallback_reason if used_fallback else "")

            st.session_state.chat_messages.append({
                "role": "assistant", "content": full_answer,
                "sources": sources,
                "fallback_note": fallback_reason if used_fallback else "",
            })

        except AppError as exc:
            placeholder.warning(exc.user_message)
            st.session_state.chat_messages.append({
                "role": "assistant", "content": exc.user_message, "sources": [],
            })
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected chat failure")
            placeholder.error("I couldn't generate an answer right now. Please try again.")
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": "I couldn't generate an answer right now. Please try again.",
                "sources": [],
            })
