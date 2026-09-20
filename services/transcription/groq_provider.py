"""
services/transcription/groq_provider.py
===============================================================================
Primary transcription provider: Groq's hosted Whisper Large V3 Turbo.

Cloud inference removes the need to download/run a local Whisper model on
CPU (previously the single largest source of latency in this project).
===============================================================================
"""

from __future__ import annotations

import time
from pathlib import Path

from groq import Groq

from models.transcript import Transcript, TranscriptSegment
from services.transcription.base import TranscriptionProvider
from utils.errors import ConfigurationError, TranscriptionError
from utils.logging import get_logger, log_stage

logger = get_logger(__name__)

_MAX_RETRIES = 2


class GroqWhisperProvider(TranscriptionProvider):
    name = "groq"

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ConfigurationError(
                "GROQ_API_KEY is not set.", detail="required for transcription"
            )
        self.model = model
        self._client = Groq(api_key=api_key)

    def transcribe(
        self, audio_path: Path, *, language: str | None = None, translate: bool = False
    ) -> Transcript:
        last_error: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            started = time.perf_counter()
            try:
                with open(audio_path, "rb") as f:
                    file_bytes = f.read()

                endpoint = (
                    self._client.audio.translations
                    if translate
                    else self._client.audio.transcriptions
                )

                kwargs = dict(
                    file=(audio_path.name, file_bytes),
                    model=self.model,
                    response_format="verbose_json",
                    temperature=0.0,
                )
                if not translate and language and language.lower() != "auto":
                    kwargs["language"] = language

                response = endpoint.create(**kwargs)

                segments = [
                    TranscriptSegment(
                        start=float(seg.get("start", 0.0)),
                        end=float(seg.get("end", 0.0)),
                        text=str(seg.get("text", "")).strip(),
                    )
                    for seg in (getattr(response, "segments", None) or [])
                ]

                transcript = Transcript(
                    text=(getattr(response, "text", "") or "").strip(),
                    language=getattr(response, "language", None) or (language or "unknown"),
                    segments=segments,
                    provider="groq",
                    model=self.model,
                    duration_seconds=getattr(response, "duration", None),
                )

                log_stage(
                    logger, "transcription",
                    duration=time.perf_counter() - started,
                    provider="groq", model=self.model, attempt=attempt,
                    words=transcript.word_count,
                )
                return transcript

            except Exception as exc:  # noqa: BLE001
                last_error = exc
                log_stage(
                    logger, "transcription",
                    duration=time.perf_counter() - started,
                    status="retry" if attempt < _MAX_RETRIES else "failed",
                    provider="groq", model=self.model, attempt=attempt,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(1.5 * attempt)

        raise TranscriptionError(
            "Groq transcription failed after retrying. "
            "The audio may be unsupported, or the service may be temporarily unavailable.",
            detail=str(last_error),
        )
