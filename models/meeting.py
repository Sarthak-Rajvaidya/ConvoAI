"""
models/meeting.py
===============================================================================
Structured output schema for the single primary LLM analysis call (see
services/analysis/meeting_analyzer.py). Every field is validated on the way
out of the LLM so the rest of the app can trust its shape.
===============================================================================
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class ActionItem(BaseModel):
    task: str = ""
    owner: str | None = None
    deadline: str | None = None
    priority: str | None = None
    status: str | None = None


class MeetingAnalysis(BaseModel):
    title: str = "Untitled Meeting"
    content_type: str = "meeting"
    language: str = "unknown"
    executive_summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    risks_and_blockers: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    sentiment: str = "neutral"
    meeting_outcome: str = ""

    @classmethod
    def fallback(cls, transcript_excerpt: str) -> "MeetingAnalysis":
        """A minimal, always-valid analysis used when the LLM's structured
        output can't be parsed/repaired. Keeps the product usable instead
        of failing the whole pipeline."""
        excerpt = " ".join(transcript_excerpt.split()[:60])
        return cls(
            title="Untitled Meeting",
            content_type="meeting",
            executive_summary=(
                "Automatic summarization was unavailable for this transcript. "
                f"Excerpt: {excerpt}..."
            ),
        )


class MeetingResult(BaseModel):
    """The full result of one processing run: transcript + analysis +
    performance metrics, kept together for session state / reports."""

    model_config = {"arbitrary_types_allowed": True}

    meeting_id: str
    source_label: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    analysis: MeetingAnalysis
    transcript_text: str = ""
    transcript_language: str = "unknown"
    transcript_provider: str = "unknown"
    duration_seconds: float | None = None
    timings: dict[str, float] = Field(default_factory=dict)
