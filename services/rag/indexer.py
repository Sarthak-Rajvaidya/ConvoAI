"""
services/rag/indexer.py
===============================================================================
Runs semantic indexing in a background thread so the user sees meeting
results immediately instead of waiting for embeddings + Qdrant upserts.

    Transcript --> chunk --> (background thread) --> embed --> Qdrant upsert
                                                              --> status: ready

If Qdrant isn't configured/reachable, indexing is marked "unavailable" and
the chat service transparently falls back to the in-memory transcript store
(see services/rag/retriever.py) -- the product stays usable either way.
===============================================================================
"""

from __future__ import annotations

import threading

from config.settings import Settings, get_settings
from models.transcript import Transcript
from services.rag.chunking import chunk_transcript
from services.rag.memory_store import chunk_memory_store, rag_status_tracker
from utils.errors import ConfigurationError
from utils.logging import get_logger, log_stage
from utils.timing import StageTimer

logger = get_logger(__name__)


def _index_worker(meeting_id: str, transcript: Transcript, settings: Settings) -> None:
    timer = StageTimer()
    rag_status_tracker.set(meeting_id, state="indexing")

    chunks = chunk_transcript(
        transcript,
        meeting_id,
        chunk_size_words=settings.rag_chunk_size_words,
        overlap_words=settings.rag_chunk_overlap_words,
    )
    # Always available immediately for transcript-based fallback retrieval.
    chunk_memory_store.put(meeting_id, chunks)

    if not settings.qdrant_configured:
        rag_status_tracker.set(
            meeting_id, state="unavailable", chunk_count=len(chunks),
            error="Qdrant is not configured; using transcript-based retrieval.",
        )
        return

    try:
        from services.rag.embeddings import Embedder
        from services.rag.vector_store import VectorStore

        with timer.stage("rag_index"):
            embedder = Embedder(settings.embedding_model)
            store = VectorStore(settings, embedder)
            count = store.index_chunks(chunks)

        rag_status_tracker.set(meeting_id, state="ready", chunk_count=count, error="")
        log_stage(
            logger, "rag_index", duration=timer.durations.get("rag_index", 0),
            meeting_id=meeting_id, chunks=count,
        )
    except ConfigurationError as exc:
        rag_status_tracker.set(
            meeting_id, state="unavailable", chunk_count=len(chunks), error=exc.user_message
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Background RAG indexing failed for %s", meeting_id)
        rag_status_tracker.set(
            meeting_id, state="error", chunk_count=len(chunks), error=str(exc)
        )


def start_background_indexing(meeting_id: str, transcript: Transcript) -> None:
    settings = get_settings()
    rag_status_tracker.set(meeting_id, state="pending", chunk_count=0, error="")
    thread = threading.Thread(
        target=_index_worker,
        args=(meeting_id, transcript, settings),
        name=f"rag-index-{meeting_id[:8]}",
        daemon=True,
    )
    thread.start()
