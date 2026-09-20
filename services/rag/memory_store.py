"""
services/rag/memory_store.py
===============================================================================
Two small, thread-safe, in-process registries:

1. `ChunkMemoryStore` -- keeps each meeting's transcript chunks in memory so
   the chatbot can answer from the transcript directly the instant analysis
   finishes, before (or if) Qdrant indexing completes. This is also the
   fallback retrieval path if Qdrant is temporarily unavailable (section 40
   of the brief).

2. `RagStatusTracker` -- tracks background indexing status per meeting_id
   ("pending" / "indexing" / "ready" / "error" / "unavailable"), updated by
   the background thread started in services/rag/indexer.py and polled by
   the UI.

Both are simple module-level singletons guarded by a lock -- deliberately
not a database, since state only needs to live for the process lifetime of
this single-user Streamlit app.
===============================================================================
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from models.rag import TranscriptChunk


class ChunkMemoryStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._chunks: dict[str, list[TranscriptChunk]] = {}

    def put(self, meeting_id: str, chunks: list[TranscriptChunk]) -> None:
        with self._lock:
            self._chunks[meeting_id] = chunks

    def get(self, meeting_id: str) -> list[TranscriptChunk]:
        with self._lock:
            return list(self._chunks.get(meeting_id, []))


@dataclass
class RagStatus:
    state: str = "pending"  # pending | indexing | ready | error | unavailable
    chunk_count: int = 0
    error: str = ""


class RagStatusTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._status: dict[str, RagStatus] = {}

    def set(self, meeting_id: str, **kwargs) -> None:
        with self._lock:
            current = self._status.get(meeting_id, RagStatus())
            for key, value in kwargs.items():
                setattr(current, key, value)
            self._status[meeting_id] = current

    def get(self, meeting_id: str) -> RagStatus:
        with self._lock:
            return self._status.get(meeting_id, RagStatus())


# Process-lifetime singletons.
chunk_memory_store = ChunkMemoryStore()
rag_status_tracker = RagStatusTracker()
