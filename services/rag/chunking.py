"""
services/rag/chunking.py
===============================================================================
Splits a Transcript into overlapping, timestamp-aware chunks suitable for
embedding + retrieval. If the transcript has segment-level timestamps (from
Groq), chunks are built by grouping consecutive segments up to a target word
count, so every chunk keeps an accurate start/end time. Falls back to plain
word-window chunking when no segments are available.
===============================================================================
"""

from __future__ import annotations

from models.rag import TranscriptChunk
from models.transcript import Transcript


def chunk_transcript(
    transcript: Transcript,
    meeting_id: str,
    *,
    chunk_size_words: int = 180,
    overlap_words: int = 30,
) -> list[TranscriptChunk]:
    if transcript.segments:
        return _chunk_from_segments(transcript, meeting_id, chunk_size_words)
    return _chunk_from_text(transcript.text, meeting_id, chunk_size_words, overlap_words)


def _chunk_from_segments(
    transcript: Transcript, meeting_id: str, chunk_size_words: int
) -> list[TranscriptChunk]:
    chunks: list[TranscriptChunk] = []
    buffer_words: list[str] = []
    buffer_start: float | None = None
    buffer_end: float | None = None
    chunk_id = 0

    def flush():
        nonlocal buffer_words, buffer_start, buffer_end, chunk_id
        if not buffer_words:
            return
        chunks.append(
            TranscriptChunk(
                meeting_id=meeting_id,
                chunk_id=chunk_id,
                text=" ".join(buffer_words).strip(),
                start=buffer_start,
                end=buffer_end,
            )
        )
        chunk_id += 1
        buffer_words = []
        buffer_start = None
        buffer_end = None

    for seg in transcript.segments:
        words = seg.text.split()
        if not words:
            continue
        if buffer_start is None:
            buffer_start = seg.start
        buffer_end = seg.end
        buffer_words.extend(words)
        if len(buffer_words) >= chunk_size_words:
            flush()

    flush()
    if not chunks and transcript.text.strip():
        return _chunk_from_text(transcript.text, meeting_id, chunk_size_words, 30)
    return chunks


def _chunk_from_text(
    text: str, meeting_id: str, chunk_size_words: int, overlap_words: int
) -> list[TranscriptChunk]:
    words = text.split()
    if not words:
        return []

    chunks: list[TranscriptChunk] = []
    step = max(1, chunk_size_words - overlap_words)
    chunk_id = 0
    for start_idx in range(0, len(words), step):
        window = words[start_idx : start_idx + chunk_size_words]
        if not window:
            continue
        chunks.append(
            TranscriptChunk(
                meeting_id=meeting_id,
                chunk_id=chunk_id,
                text=" ".join(window),
            )
        )
        chunk_id += 1
        if start_idx + chunk_size_words >= len(words):
            break
    return chunks
