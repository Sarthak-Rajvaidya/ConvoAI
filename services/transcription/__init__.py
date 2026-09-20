"""
services/transcription/__init__.py
===============================================================================
Factory for building the configured transcription provider, plus a helper
that runs transcription across one or more audio chunks (in parallel when
there's more than one) and merges the results into a single Transcript with
correctly offset timestamps.
===============================================================================
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config.settings import Settings
from models.transcript import Transcript, TranscriptSegment
from services.transcription.base import TranscriptionProvider
from utils.errors import ConfigurationError
from utils.logging import get_logger

logger = get_logger(__name__)


def get_transcription_provider(settings: Settings) -> TranscriptionProvider:
    provider = settings.transcription_provider

    if provider == "groq":
        from services.transcription.groq_provider import GroqWhisperProvider
        return GroqWhisperProvider(settings.groq_api_key, settings.groq_transcription_model)

    if provider == "local":
        from services.transcription.local_provider import LocalWhisperProvider
        return LocalWhisperProvider(settings.whisper_local_model)

    raise ConfigurationError(
        f"Unknown TRANSCRIPTION_PROVIDER '{provider}'. Use 'groq' or 'local'."
    )


def transcribe_chunks(
    provider: TranscriptionProvider,
    chunks: list[tuple[Path, float]],
    *,
    language: str | None,
    translate: bool,
    max_workers: int,
) -> Transcript:
    """Transcribe one or more (path, start_offset) chunks and merge them
    into a single Transcript, in original chunk order, with segment
    timestamps shifted by each chunk's start offset."""

    if len(chunks) == 1:
        path, offset = chunks[0]
        transcript = provider.transcribe(path, language=language, translate=translate)
        return _shift(transcript, offset)

    results: dict[int, Transcript] = {}
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {
            pool.submit(provider.transcribe, path, language=language, translate=translate): (
                idx, offset
            )
            for idx, (path, offset) in enumerate(chunks)
        }
        for future in as_completed(futures):
            idx, offset = futures[future]
            transcript = future.result()
            results[idx] = _shift(transcript, offset)

    ordered = [results[i] for i in sorted(results)]
    merged_text = " ".join(t.text for t in ordered if t.text).strip()
    merged_segments: list[TranscriptSegment] = []
    for t in ordered:
        merged_segments.extend(t.segments)

    languages = [t.language for t in ordered if t.language and t.language != "unknown"]
    language_out = languages[0] if languages else "unknown"

    return Transcript(
        text=merged_text,
        language=language_out,
        segments=merged_segments,
        provider=ordered[0].provider if ordered else "unknown",
        model=ordered[0].model if ordered else "unknown",
        duration_seconds=sum(t.duration_seconds or 0 for t in ordered) or None,
    )


def _shift(transcript: Transcript, offset: float) -> Transcript:
    if not offset:
        return transcript
    shifted = [
        TranscriptSegment(start=seg.start + offset, end=seg.end + offset, text=seg.text)
        for seg in transcript.segments
    ]
    return transcript.model_copy(update={"segments": shifted})
