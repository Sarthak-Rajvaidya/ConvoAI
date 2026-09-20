"""
services/rag/vector_store.py
===============================================================================
Qdrant Cloud wrapper: collection bootstrap (once per process), batched
upserts with deterministic point IDs, and meeting-scoped similarity search.

Meeting isolation is enforced everywhere: every query is filtered by
`meeting_id`, and there's a keyword payload index on it so Qdrant Cloud
accepts filtered queries efficiently and one meeting's chunks can never
leak into another meeting's answers.
===============================================================================
"""

from __future__ import annotations

import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from config.settings import Settings
from models.rag import SourceChunk, TranscriptChunk
from services.rag.embeddings import Embedder
from utils.errors import ConfigurationError, RAGIndexError, RAGRetrievalError
from utils.logging import get_logger

logger = get_logger(__name__)

_ensured_collections: set[str] = set()


@lru_cache(maxsize=1)
def _get_client(url: str, api_key: str) -> QdrantClient:
    if not url or not api_key:
        raise ConfigurationError("QDRANT_URL / QDRANT_API_KEY are not set.")
    return QdrantClient(url=url, api_key=api_key)


class VectorStore:
    def __init__(self, settings: Settings, embedder: Embedder):
        self.settings = settings
        self.embedder = embedder
        self.collection = settings.qdrant_collection
        self._client = _get_client(settings.qdrant_url, settings.qdrant_api_key)

    def _ensure_collection(self) -> None:
        if self.collection in _ensured_collections:
            return
        try:
            if not self._client.collection_exists(self.collection):
                self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=self.embedder.dimension, distance=Distance.COSINE
                    ),
                )
                logger.info("Created Qdrant collection: %s", self.collection)

            self._client.create_payload_index(
                collection_name=self.collection,
                field_name="meeting_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            if "already exists" not in message and "duplicate" not in message:
                raise RAGIndexError(
                    "Couldn't prepare the Qdrant collection.", detail=str(exc)
                ) from exc
        _ensured_collections.add(self.collection)

    def index_chunks(self, chunks: list[TranscriptChunk]) -> int:
        if not chunks:
            return 0
        self._ensure_collection()

        vectors = self.embedder.embed_documents([c.text for c in chunks])
        points = []
        for chunk, vector in zip(chunks, vectors):
            point_id = str(
                uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk.meeting_id}-{chunk.chunk_id}")
            )
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "meeting_id": chunk.meeting_id,
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "start": chunk.start,
                        "end": chunk.end,
                        "source": chunk.source,
                    },
                )
            )

        try:
            # Batch upsert -- one round trip per 64 points is enough to stay
            # fast without hammering the API for very long meetings.
            batch_size = 64
            for i in range(0, len(points), batch_size):
                self._client.upsert(
                    collection_name=self.collection, points=points[i : i + batch_size]
                )
        except Exception as exc:  # noqa: BLE001
            raise RAGIndexError(
                "Couldn't write transcript chunks to Qdrant.", detail=str(exc)
            ) from exc

        return len(points)

    def search(self, query: str, meeting_id: str, top_k: int) -> list[SourceChunk]:
        self._ensure_collection()
        vector = self.embedder.embed_query(query)

        try:
            results = self._client.query_points(
                collection_name=self.collection,
                query=vector,
                query_filter=Filter(
                    must=[FieldCondition(key="meeting_id", match=MatchValue(value=meeting_id))]
                ),
                limit=top_k,
                with_payload=True,
            ).points
        except Exception as exc:  # noqa: BLE001
            raise RAGRetrievalError(
                "Semantic search is temporarily unavailable.", detail=str(exc)
            ) from exc

        return [
            SourceChunk(
                chunk_id=point.payload.get("chunk_id", 0),
                text=point.payload.get("text", ""),
                start=point.payload.get("start"),
                end=point.payload.get("end"),
                score=float(point.score),
            )
            for point in results
        ]
