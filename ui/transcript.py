"""
ui/transcript.py
===============================================================================
Transcript viewer with keyword search and (when available) timestamped
results.
===============================================================================
"""

from __future__ import annotations

import re

import streamlit as st

from models.meeting import MeetingResult
from services.rag.memory_store import chunk_memory_store
from ui.components import empty_state, section_label


def render_transcript_page() -> None:
    result: MeetingResult | None = st.session_state.get("result")
    if result is None:
        empty_state("📝", "No transcript yet", "Analyze a meeting first.")
        return

    section_label("Search transcript")
    query = st.text_input("Search transcript...", key="transcript_search", label_visibility="collapsed",
                           placeholder="Search transcript...")

    chunks = chunk_memory_store.get(result.meeting_id)

    if query and query.strip():
        pattern = re.compile(re.escape(query.strip()), re.IGNORECASE)
        matches = [c for c in chunks if pattern.search(c.text)]
        if not matches:
            st.caption("No matches found.")
        for c in matches:
            highlighted = pattern.sub(lambda m: f"**:orange[{m.group(0)}]**", c.text)
            ts = _format_ts(c.start) if c.start is not None else None
            label = f"`{ts}`  " if ts else ""
            st.markdown(f"{label}{highlighted}")
            st.divider()
    else:
        section_label("Full transcript")
        if chunks:
            for c in chunks:
                ts = _format_ts(c.start) if c.start is not None else None
                label = f"**{ts}**  " if ts else ""
                st.markdown(f"{label}{c.text}")
        else:
            st.write(result.transcript_text)


def _format_ts(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
