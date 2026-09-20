"""
services/rag/retriever.py
===============================================================================
Practical hybrid retrieval:

  1. Semantic search in Qdrant (when indexing has completed for this
     meeting).
  2. Lexical fallback: simple keyword-overlap scoring over the meeting's
     transcript chunks (from the in-memory store), used when Qdrant isn't
     ready yet, isn't configured, or a query is looking for something like
     an exact name/date/ID that embeddings sometimes miss.
  3. Results from both are merged, deduplicated by chunk_id, and filtered
     by a minimum relevance threshold. Meeting isolation is guaranteed
     because both paths are always scoped to one meeting_id.
===============================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from config.settings import Settings
from models.rag import SourceChunk, TranscriptChunk
from services.rag.memory_store import chunk_memory_store, rag_status_tracker
from utils.logging import get_logger

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def lexical_search(query: str, chunks: list[TranscriptChunk], top_k: int) -> list[SourceChunk]:
    query_tokens = _tokenize(query)
    if not query_tokens or not chunks:
        return []

    scored: list[SourceChunk] = []
    for chunk in chunks:
        chunk_tokens = _tokenize(chunk.text)
        if not chunk_tokens:
            continue
        overlap = len(query_tokens & chunk_tokens)
        if overlap == 0:
            continue
        score = overlap / max(1, len(query_tokens))
        scored.append(
            SourceChunk(
                chunk_id=chunk.chunk_id, text=chunk.text,
                start=chunk.start, end=chunk.end, score=round(score, 4),
            )
        )

    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:top_k]


@dataclass
class RetrievalResult:
    sources: list[SourceChunk]
    used_fallback: bool
    fallback_reason: str = ""


def retrieve(query: str, meeting_id: str, settings: Settings) -> RetrievalResult:
    status = rag_status_tracker.get(meeting_id)
    memory_chunks = chunk_memory_store.get(meeting_id)

    semantic_results: list[SourceChunk] = []
    used_fallback = False
    fallback_reason = ""

    if status.state == "ready" and settings.qdrant_configured:
        try:
            from services.rag.embeddings import Embedder
            from services.rag.vector_store import VectorStore

            embedder = Embedder(settings.embedding_model)
            store = VectorStore(settings, embedder)
            semantic_results = store.search(query, meeting_id, settings.rag_top_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Semantic search failed, falling back to transcript search: %s", exc)
            used_fallback = True
            fallback_reason = "Semantic search is temporarily unavailable."
    else:
        used_fallback = True
        fallback_reason = {
            "pending": "Knowledge base is still preparing.",
            "indexing": "Knowledge base is still preparing.",
            "unavailable": "Semantic search isn't configured for this deployment.",
            "error": "Semantic search failed to build for this meeting.",
        }.get(status.state, "Using transcript-based search.")

    # Always compute lexical results too (cheap) and merge -- catches exact
    # names/dates/IDs that pure embeddings sometimes rank low.
    lexical_results = lexical_search(query, memory_chunks, settings.rag_top_k)

    merged: dict[int, SourceChunk] = {}
    for chunk in semantic_results + lexical_results:
        existing = merged.get(chunk.chunk_id)
        if existing is None or chunk.score > existing.score:
            merged[chunk.chunk_id] = chunk

    ranked = sorted(merged.values(), key=lambda c: c.score, reverse=True)

    min_score = settings.rag_min_score if semantic_results else 0.08
    filtered = [c for c in ranked if c.score >= min_score][: settings.rag_top_k]

    return RetrievalResult(
        sources=filtered, used_fallback=used_fallback, fallback_reason=fallback_reason
    )
