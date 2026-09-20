"""
ui/components.py
===============================================================================
Reusable UI building blocks used across the app's pages: MetricCard,
StatusBadge, ProcessingTimeline, InsightCard, ActionItemTable,
SourceCitation, ChatMessage helpers, EmptyState, ErrorState.
===============================================================================
"""

from __future__ import annotations

import streamlit as st

from models.meeting import MeetingAnalysis
from models.rag import SourceChunk


# =============================================================================
# Badges
# =============================================================================

def status_badge(label: str, kind: str = "neutral") -> str:
    return f'<span class="badge badge-{kind}">{label}</span>'


def render_badges(*badges: tuple[str, str]) -> None:
    html = "".join(status_badge(label, kind) for label, kind in badges)
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# Metric tiles
# =============================================================================

def metric_row(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(
                f'<div class="metric-tile"><div class="m-value">{value}</div>'
                f'<div class="m-label">{label}</div></div>',
                unsafe_allow_html=True,
            )


# =============================================================================
# Empty / error states
# =============================================================================

def empty_state(icon: str, title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="empty-state"><div class="e-icon">{icon}</div>'
        f'<div style="font-weight:600;color:inherit;">{title}</div>'
        f'<div class="muted">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def error_state(title: str, detail: str = "") -> None:
    st.error(f"**{title}**" + (f"\n\n{detail}" if detail else ""))


# =============================================================================
# Processing timeline
# =============================================================================

TIMELINE_STAGES = [
    ("audio", "Audio acquired"),
    ("transcription", "Speech transcribed"),
    ("analysis", "Meeting analyzed"),
    ("indexing", "Knowledge base preparing"),
]


def render_timeline(current_stage: str, completed: set[str]) -> None:
    stage_keys = [key for key, _ in TIMELINE_STAGES]
    reached = current_stage in stage_keys

    lines = []
    for key, label in TIMELINE_STAGES:
        if key in completed:
            icon = "✅"
        elif key == current_stage:
            icon = "🔄"
        else:
            icon = "○"
        lines.append(f'<div class="timeline-item"><span class="timeline-dot">{icon}</span>{label}</div>')
    st.markdown("".join(lines), unsafe_allow_html=True)


# =============================================================================
# Insight / summary cards
# =============================================================================

def card(html_body: str) -> None:
    st.markdown(f'<div class="card">{html_body}</div>', unsafe_allow_html=True)


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


# =============================================================================
# Action items table
# =============================================================================

def render_action_items_table(analysis: MeetingAnalysis) -> None:
    if not analysis.action_items:
        empty_state("✅", "No action items detected", "This meeting didn't surface any tracked tasks.")
        return

    priority_kind = {"high": "danger", "medium": "warning", "low": "neutral"}
    rows = []
    for item in analysis.action_items:
        rows.append({
            "Task": item.task,
            "Owner": item.owner or "Unassigned",
            "Deadline": item.deadline or "—",
            "Priority": (item.priority or "—").title(),
            "Status": (item.status or "pending").title(),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


# =============================================================================
# Source citations (RAG)
# =============================================================================

def render_sources(sources: list[SourceChunk], fallback_note: str = "") -> None:
    if fallback_note:
        st.caption(f"ℹ️ {fallback_note}")
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            ts = s.timestamp_label()
            meta = f"Chunk {s.chunk_id}" + (f" · {ts}" if ts else "") + f" · relevance {s.score:.2f}"
            excerpt = s.text if len(s.text) < 320 else s.text[:320] + "…"
            st.markdown(
                f'<div class="source-chip"><div class="s-meta">{meta}</div>{excerpt}</div>',
                unsafe_allow_html=True,
            )


# =============================================================================
# Feature cards (landing page)
# =============================================================================

def render_feature_grid(features: list[tuple[str, str, str]]) -> None:
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(
                f'<div class="feature-card"><div class="f-title">{icon} {title}</div>'
                f'<div class="f-desc">{desc}</div></div>',
                unsafe_allow_html=True,
            )
