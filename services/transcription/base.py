"""
services/transcription/base.py
===============================================================================
Abstraction over speech-to-text providers.

    TranscriptionProvider
            |
            +-- GroqWhisperProvider   (default, cloud, fast)
            +-- LocalWhisperProvider  (optional fallback, offline)
            +-- ... future providers
===============================================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from models.transcript import Transcript


class TranscriptionProvider(ABC):
    name: str = "base"

    @abstractmethod
    def transcribe(
        self, audio_path: Path, *, language: str | None = None, translate: bool = False
    ) -> Transcript:
        """Transcribe a single audio file. `language` is an optional ISO
        hint ('en', 'hi', ...) or None/'auto' for auto-detection.
        `translate=True` requests an English translation instead of
        transcription in the original language."""
        raise NotImplementedError
