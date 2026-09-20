"""
services/pipeline.py
===============================================================================
Orchestrates the end-to-end run:

    source (YouTube URL or uploaded file)
        |
        v
    audio acquisition (compressed, no forced WAV re-encode)
        |
        v
    Groq Whisper transcription (parallel chunks only if the file is large)
        |
        v
    ONE structured Groq analysis call
        |
        v
    MeetingResult returned immediately  <-- UI shows this right away
        |
        v
    RAG indexing kicked off in a background thread (non-blocking)

Every stage is timed and logged via utils.timing / utils.logging.
===============================================================================
"""

from __future__ import annotations

import uuid
from pathlib import Path

from config.settings import Settings
from models.meeting import MeetingResult
from models.transcript import Transcript
from services.analysis.meeting_analyzer import MeetingAnalyzer
from services.audio import processor as audio_processor
from services.rag.indexer import start_background_indexing
from services.transcription import get_transcription_provider, transcribe_chunks
from utils.errors import AppError
from utils.logging import get_logger, log_stage
from utils.timing import StageTimer

logger = get_logger(__name__)


def run_pipeline(
    source: str,
    *,
    is_youtube: bool,
    uploaded_tmp_path: Path | None,
    uploaded_filename: str | None,
    uploaded_size: int | None,
    settings: Settings,
    language_hint: str | None,
    translate_to_english: bool,
    progress_callback=None,
) -> MeetingResult:
    """Runs the full synchronous pipeline (through analysis) and kicks off
    background RAG indexing before returning. `progress_callback(stage_key)`
    is called at the start of each stage so the UI can update a timeline."""

    def report(stage: str):
        if progress_callback:
            progress_callback(stage)

    timer = StageTimer()
    meeting_id = uuid.uuid4().hex

    # -- 1. Audio acquisition -------------------------------------------------
    report("audio")
    with timer.stage("audio_acquisition"):
        if is_youtube:
            acquired = audio_processor.download_youtube_audio(
                source, audio_processor.new_work_dir()
            )
        else:
            acquired = audio_processor.acquire_uploaded_file(
                uploaded_tmp_path, uploaded_filename, uploaded_size, settings
            )

    try:
        # -- 2. Transcription ---------------------------------------------------
        report("transcription")
        with timer.stage("transcription"):
            provider = get_transcription_provider(settings)

            if audio_processor.needs_segmentation(acquired, settings.max_audio_segment_mb):
                chunk_pairs = audio_processor.segment_audio(acquired, settings.audio_chunk_seconds)
            else:
                chunk_pairs = [(acquired.path, 0.0)]

            transcript: Transcript = transcribe_chunks(
                provider, chunk_pairs,
                language=language_hint, translate=translate_to_english,
                max_workers=settings.max_parallel_transcriptions,
            )

        if not transcript.text.strip():
            raise AppError("Transcription returned no speech. Try a different source.")

        # -- 3. Analysis (single structured call) --------------------------------
        report("analysis")
        with timer.stage("analysis"):
            analyzer = MeetingAnalyzer(settings)
            analysis = analyzer.analyze(transcript.text)

    finally:
        audio_processor.cleanup(acquired)

    result = MeetingResult(
        meeting_id=meeting_id,
        source_label=acquired.label,
        analysis=analysis,
        transcript_text=transcript.text,
        transcript_language=transcript.language,
        transcript_provider=transcript.provider,
        duration_seconds=acquired.duration_seconds,
        timings=timer.as_dict(),
    )

    log_stage(
        logger, "pipeline", duration=timer.total, meeting_id=meeting_id,
        words=transcript.word_count, provider=transcript.provider,
    )

    # -- 4. RAG indexing in the background -- do NOT block the return ---------
    report("indexing")
    start_background_indexing(meeting_id, transcript)

    return result
