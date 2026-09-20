"""
models/rag.py
===============================================================================
Models shared by the RAG indexing/retrieval/chat services.
===============================================================================
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TranscriptChunk(BaseModel):
    """One indexable unit of a transcript, kept in both Qdrant payloads and
    the in-memory fallback store."""

    meeting_id: str
    chunk_id: int
    text: str
    start: float | None = None
    end: float | None = None
    source: str = "meeting_transcript"


class SourceChunk(BaseModel):
    """A retrieved chunk, annotated with a relevance score, returned to the
    UI for citation display."""

    chunk_id: int
    text: str
    start: float | None = None
    end: float | None = None
    score: float = 0.0

    def timestamp_label(self) -> str | None:
        if self.start is None:
            return None
        seconds = max(0, int(self.start))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class ChatTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatAnswer(BaseModel):
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    used_fallback: bool = False
