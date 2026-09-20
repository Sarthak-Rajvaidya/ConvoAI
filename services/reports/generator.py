"""
services/reports/generator.py
===============================================================================
Export a MeetingResult as Markdown, TXT, JSON, a plain transcript file, or a
formatted PDF report (via reportlab).
===============================================================================
"""

from __future__ import annotations

import io
import json

from models.meeting import MeetingResult


def generate_markdown_report(result: MeetingResult) -> str:
    a = result.analysis
    lines = [f"# {a.title}", ""]
    lines += [
        "## Meeting Info", "",
        f"- **Type:** {a.content_type}",
        f"- **Language:** {a.language}",
        f"- **Generated:** {result.created_at}",
        f"- **Sentiment:** {a.sentiment}",
        "",
        "## Executive Summary", "",
        a.executive_summary or "_No summary available._", "",
    ]

    if a.key_points:
        lines += ["## Key Points", "", *(f"- {p}" for p in a.key_points), ""]

    if a.decisions:
        lines += ["## Decisions", "", *(f"{i}. {d}" for i, d in enumerate(a.decisions, 1)), ""]

    if a.action_items:
        lines += ["## Action Items", "", "| Task | Owner | Deadline | Priority | Status |",
                   "|---|---|---|---|---|"]
        for item in a.action_items:
            lines.append(
                f"| {item.task} | {item.owner or '—'} | {item.deadline or '—'} | "
                f"{item.priority or '—'} | {item.status or '—'} |"
            )
        lines.append("")

    if a.risks_and_blockers:
        lines += ["## Risks & Blockers", "", *(f"- {r}" for r in a.risks_and_blockers), ""]

    if a.open_questions:
        lines += ["## Open Questions", "", *(f"- {q}" for q in a.open_questions), ""]

    if a.topics:
        lines += ["## Topics", "", ", ".join(a.topics), ""]

    if a.next_steps:
        lines += ["## Next Steps", "", *(f"- [ ] {s}" for s in a.next_steps), ""]

    lines += ["## Meeting Outcome", "", a.meeting_outcome or "_Not available._", ""]
    return "\n".join(lines)


def generate_txt_report(result: MeetingResult) -> str:
    md = generate_markdown_report(result)
    text = md.replace("## ", "").replace("# ", "").replace("**", "").replace("- [ ] ", "- ")
    return text


def generate_json_report(result: MeetingResult) -> str:
    return json.dumps(result.model_dump(), indent=2, ensure_ascii=False)


def generate_transcript_export(result: MeetingResult) -> str:
    header = f"Transcript — {result.analysis.title}\nGenerated: {result.created_at}\n\n"
    return header + result.transcript_text


def generate_pdf_report(result: MeetingResult) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    a = result.analysis
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=LETTER,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontSize=20, spaceAfter=4,
    )
    meta_style = ParagraphStyle(
        "Meta", parent=styles["Normal"], textColor=colors.HexColor("#6b7280"), fontSize=9,
    )
    heading_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], spaceBefore=16, spaceAfter=6,
        textColor=colors.HexColor("#111827"),
    )
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.5, leading=15)

    story = [
        Paragraph(a.title, title_style),
        Paragraph(
            f"{a.content_type.title()} · {a.language} · Generated {result.created_at[:19]}",
            meta_style,
        ),
        Spacer(1, 12),
        Paragraph("Executive Summary", heading_style),
        Paragraph(a.executive_summary or "No summary available.", body_style),
    ]

    def bullet_section(title: str, items: list[str]):
        if not items:
            return
        story.append(Paragraph(title, heading_style))
        story.append(
            ListFlowable(
                [ListItem(Paragraph(item, body_style)) for item in items],
                bulletType="bullet",
            )
        )

    bullet_section("Key Points", a.key_points)
    bullet_section("Decisions", a.decisions)

    if a.action_items:
        story.append(Paragraph("Action Items", heading_style))
        table_data = [["Task", "Owner", "Deadline", "Priority", "Status"]]
        for item in a.action_items:
            table_data.append([
                item.task, item.owner or "—", item.deadline or "—",
                item.priority or "—", item.status or "—",
            ])
        table = Table(table_data, colWidths=[1.9 * inch, 0.9 * inch, 0.9 * inch, 0.8 * inch, 0.8 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
        ]))
        story.append(table)

    bullet_section("Risks & Blockers", a.risks_and_blockers)
    bullet_section("Open Questions", a.open_questions)
    bullet_section("Next Steps", a.next_steps)

    if a.topics:
        story.append(Paragraph("Topics", heading_style))
        story.append(Paragraph(", ".join(a.topics), body_style))

    story.append(Paragraph("Meeting Outcome", heading_style))
    story.append(Paragraph(a.meeting_outcome or "Not available.", body_style))

    story.append(Paragraph("Transcript Reference", heading_style))
    excerpt = result.transcript_text[:1200]
    suffix = "…" if len(result.transcript_text) > 1200 else ""
    story.append(Paragraph((excerpt + suffix).replace("\n", "<br/>"), body_style))

    doc.build(story)
    return buffer.getvalue()
