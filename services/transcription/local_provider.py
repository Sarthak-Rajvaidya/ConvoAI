"""
services/transcription/local_provider.py
===============================================================================
Optional, offline fallback provider using `openai-whisper` on CPU/GPU.

Not installed by default (it's a heavy dependency: torch + model weights).
Install with `pip install openai-whisper` and set
`TRANSCRIPTION_PROVIDER=local` to use it. This exists so the app still has a
path to transcribe audio if Groq is unavailable or a fully offline demo is
needed -- it is intentionally NOT the primary path anymore.
===============================================================================
"""

from __future__ import annotations

import time
from pathlib import Path

from models.transcript import Transcript, TranscriptSegment
from services.transcription.base import TranscriptionProvider
from utils.errors import ConfigurationError, TranscriptionError
from utils.logging import get_logger, log_stage

logger = get_logger(__name__)


class LocalWhisperProvider(TranscriptionProvider):
    name = "local"

    def __init__(self, model_size: str = "small"):
        try:
            import whisper  # type: ignore
        except ImportError as exc:
            raise ConfigurationError(
                "Local Whisper fallback isn't installed. "
                "Run `pip install openai-whisper` or use TRANSCRIPTION_PROVIDER=groq.",
                detail=str(exc),
            ) from exc

        self._whisper = whisper
        self.model_size = model_size
        self._model = whisper.load_model(model_size)

    def transcribe(
        self, audio_path: Path, *, language: str | None = None, translate: bool = False
    ) -> Transcript:
        started = time.perf_counter()
        try:
            result = self._model.transcribe(
                str(audio_path),
                language=None if not language or language.lower() == "auto" else language,
                task="translate" if translate else "transcribe",
                fp16=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise TranscriptionError(
                "Local Whisper transcription failed.", detail=str(exc)
            ) from exc

        segments = [
            TranscriptSegment(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=str(seg.get("text", "")).strip(),
            )
            for seg in result.get("segments", [])
        ]

        transcript = Transcript(
            text=(result.get("text") or "").strip(),
            language=result.get("language", "unknown"),
            segments=segments,
            provider="local-whisper",
            model=self.model_size,
        )

        log_stage(
            logger, "transcription",
            duration=time.perf_counter() - started,
            provider="local-whisper", model=self.model_size,
            words=transcript.word_count,
        )
        return transcript
