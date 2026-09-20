"""
models/transcript.py
===============================================================================
Transcript-related data models, shared by every transcription provider.
===============================================================================
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """One timestamped chunk of speech, as returned by the STT provider."""

    start: float = 0.0
    end: float = 0.0
    text: str = ""


class Transcript(BaseModel):
    text: str = ""
    language: str = "unknown"
    segments: list[TranscriptSegment] = Field(default_factory=list)
    provider: str = "unknown"
    model: str = "unknown"
    duration_seconds: float | None = None

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def format_timestamp(self, seconds: float) -> str:
        seconds = max(0, int(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
