"""
ui/results.py
===============================================================================
The results dashboard: metrics, executive summary, decisions, action items,
risks, topics, open questions, next steps, transcript search, and exports.
===============================================================================
"""

from __future__ import annotations

import streamlit as st

from config.settings import get_settings
from models.meeting import MeetingResult
from services.rag.memory_store import rag_status_tracker
from services.reports import generator as reports
from ui.components import (
    card, empty_state, metric_row, render_action_items_table, render_badges, section_label,
)


def _rag_badge(meeting_id: str) -> tuple[str, str]:
    status = rag_status_tracker.get(meeting_id)
    return {
        "pending": ("Knowledge Base: Preparing…", "neutral"),
        "indexing": ("Knowledge Base: Indexing…", "warning"),
        "ready": ("RAG Ready", "success"),
        "unavailable": ("RAG Unavailable — using transcript search", "neutral"),
        "error": ("RAG Indexing Failed", "danger"),
    }.get(status.state, ("Knowledge Base: Preparing…", "neutral"))


def render_results_page() -> None:
    result: MeetingResult | None = st.session_state.get("result")
    if result is None:
        empty_state("📊", "No analysis yet", "Go to Analyze to process a meeting recording.")
        return

    a = result.analysis
    st.markdown(f'<div class="app-title">{a.title}</div>', unsafe_allow_html=True)

    rag_label, rag_kind = _rag_badge(result.meeting_id)
    render_badges(
        (a.content_type.title(), "accent"),
        (a.language.upper(), "neutral"),
        (rag_label, rag_kind),
    )

    duration_label = f"{int(result.duration_seconds // 60)} min" if result.duration_seconds else "—"
    word_count = len(result.transcript_text.split())
    total_time = result.timings.get("total")
    time_label = f"{total_time:.1f}s" if total_time else "—"

    metric_row([
        ("Duration", duration_label),
        ("Words", f"{word_count:,}"),
        ("Topics", str(len(a.topics))),
        ("Action Items", str(len(a.action_items))),
    ])
    st.caption(f"Processing completed in {time_label} · transcribed with {result.transcript_provider}")

    tabs = st.tabs([
        "📝 Summary", "✅ Action Items", "📌 Decisions", "⚠️ Risks",
        "❓ Questions", "🏷️ Topics", "📈 Insights", "⬇️ Downloads",
    ])

    with tabs[0]:
        section_label("Executive Summary")
        card(a.executive_summary or "<em>No summary available.</em>")
        if a.key_points:
            section_label("Key Discussion Points")
            for p in a.key_points:
                st.markdown(f"- {p}")
        if a.next_steps:
            section_label("Next Steps")
            for s in a.next_steps:
                st.checkbox(s, key=f"next_step_{hash(s)}", value=False)
        section_label("Outcome")
        st.write(a.meeting_outcome or "—")

    with tabs[1]:
        render_action_items_table(a)

    with tabs[2]:
        if a.decisions:
            for i, d in enumerate(a.decisions, 1):
                card(f"<b>Decision {i}</b><br/>{d}")
        else:
            empty_state("📌", "No decisions recorded")

    with tabs[3]:
        if a.risks_and_blockers:
            for r in a.risks_and_blockers:
                card(f"⚠️ {r}")
        else:
            empty_state("✅", "No risks or blockers flagged")

    with tabs[4]:
        if a.open_questions:
            for q in a.open_questions:
                card(f"❓ {q}")
        else:
            empty_state("❓", "No open questions")

    with tabs[5]:
        if a.topics:
            st.markdown("".join(f'<span class="pill">{t}</span>' for t in a.topics), unsafe_allow_html=True)
        else:
            empty_state("🏷️", "No topics extracted")

    with tabs[6]:
        _render_insights(a)

    with tabs[7]:
        _render_downloads(result)


def _render_insights(a) -> None:
    section_label("Meeting Health")
    metric_row([
        ("Decisions", str(len(a.decisions))),
        ("Action Items", str(len(a.action_items))),
        ("Open Questions", str(len(a.open_questions))),
        ("Risks", str(len(a.risks_and_blockers))),
    ])

    if a.action_items:
        section_label("Action Item Status")
        status_counts: dict[str, int] = {}
        for item in a.action_items:
            key = (item.status or "pending").title()
            status_counts[key] = status_counts.get(key, 0) + 1
        st.bar_chart(status_counts)

    section_label("Sentiment")
    sentiment_kind = {"positive": "success", "negative": "danger", "mixed": "warning"}.get(
        a.sentiment.lower(), "neutral"
    )
    render_badges((a.sentiment.title(), sentiment_kind))

    if a.topics:
        section_label("Discussion Focus")
        st.markdown("".join(f'<span class="pill">{t}</span>' for t in a.topics[:10]), unsafe_allow_html=True)


def _render_downloads(result: MeetingResult) -> None:
    section_label("Export this meeting")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.download_button(
            "⬇️ Markdown", reports.generate_markdown_report(result),
            file_name="meeting_report.md", mime="text/markdown", use_container_width=True,
        )
    with col2:
        st.download_button(
            "⬇️ TXT", reports.generate_txt_report(result),
            file_name="meeting_report.txt", mime="text/plain", use_container_width=True,
        )
    with col3:
        st.download_button(
            "⬇️ JSON", reports.generate_json_report(result),
            file_name="meeting_analysis.json", mime="application/json", use_container_width=True,
        )
    with col4:
        st.download_button(
            "⬇️ Transcript", reports.generate_transcript_export(result),
            file_name="transcript.txt", mime="text/plain", use_container_width=True,
        )
    with col5:
        try:
            pdf_bytes = reports.generate_pdf_report(result)
            st.download_button(
                "⬇️ PDF", pdf_bytes, file_name="meeting_report.pdf",
                mime="application/pdf", use_container_width=True,
            )
        except Exception as exc:  # noqa: BLE001
            st.caption(f"PDF export unavailable: {exc}")
