"""
config/settings.py
===============================================================================
Single source of truth for configuration. Everything comes from environment
variables (loaded from `.env` via python-dotenv), with sane defaults so the
app can start and explain what's missing rather than crash.

Nothing here ever prints/logs secret values.
===============================================================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or not val.strip():
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or not val.strip():
        return default
    try:
        return float(val)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # -- Groq (transcription + chat) -----------------------------------------
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_chat_model: str = field(
        default_factory=lambda: os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-20b")
    )
    groq_chat_model_high_quality: str = field(
        default_factory=lambda: os.getenv(
            "GROQ_CHAT_MODEL_HIGH_QUALITY", "openai/gpt-oss-120b"
        )
    )
    groq_transcription_model: str = field(
        default_factory=lambda: os.getenv(
            "GROQ_TRANSCRIPTION_MODEL", "whisper-large-v3-turbo"
        )
    )

    # -- Transcription ---------------------------------------------------------
    transcription_provider: str = field(
        default_factory=lambda: os.getenv("TRANSCRIPTION_PROVIDER", "groq").lower()
    )
    whisper_local_model: str = field(
        default_factory=lambda: os.getenv("WHISPER_LOCAL_MODEL", "small")
    )
    max_audio_segment_mb: int = field(
        default_factory=lambda: _int("MAX_AUDIO_SEGMENT_MB", 24)
    )
    audio_chunk_seconds: int = field(
        default_factory=lambda: _int("AUDIO_CHUNK_SECONDS", 600)
    )
    max_parallel_transcriptions: int = field(
        default_factory=lambda: _int("MAX_PARALLEL_TRANSCRIPTIONS", 4)
    )

    # -- Qdrant / RAG ------------------------------------------------------------
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", ""))
    qdrant_api_key: str = field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    qdrant_collection: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION", "meeting_transcripts")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    rag_top_k: int = field(default_factory=lambda: _int("RAG_TOP_K", 5))
    rag_min_score: float = field(default_factory=lambda: _float("RAG_MIN_SCORE", 0.22))
    rag_chunk_size_words: int = field(
        default_factory=lambda: _int("RAG_CHUNK_SIZE_WORDS", 180)
    )
    rag_chunk_overlap_words: int = field(
        default_factory=lambda: _int("RAG_CHUNK_OVERLAP_WORDS", 30)
    )
    chat_history_turns: int = field(
        default_factory=lambda: _int("CHAT_HISTORY_TURNS", 6)
    )

    # -- Uploads / limits --------------------------------------------------------
    max_upload_mb: int = field(default_factory=lambda: _int("MAX_UPLOAD_MB", 300))

    # -- App behaviour -------------------------------------------------------------
    demo_mode: bool = field(default_factory=lambda: _bool("DEMO_MODE", False))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key.strip())

    @property
    def qdrant_configured(self) -> bool:
        return bool(self.qdrant_url.strip() and self.qdrant_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton. Re-read by restarting the process."""
    return Settings()
