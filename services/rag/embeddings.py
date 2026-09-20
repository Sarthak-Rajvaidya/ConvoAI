"""
services/rag/embeddings.py
===============================================================================
Lightweight, production-friendly embeddings using FastEmbed (ONNX runtime --
no torch/CUDA dependency, loads in a couple of seconds, runs fine on CPU).

The model is loaded once per process (module-level singleton, also wrapped
with st.cache_resource by the caller in ui/) rather than being recreated on
every query, which was a real source of latency in the previous version.
===============================================================================
"""

from __future__ import annotations

from functools import lru_cache

from utils.errors import ConfigurationError
from utils.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_model(model_name: str):
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise ConfigurationError(
            "fastembed isn't installed. Run `pip install fastembed`.",
            detail=str(exc),
        ) from exc

    logger.info("Loading embedding model %s (first call only)", model_name)
    return TextEmbedding(model_name=model_name)


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = _get_model(model_name)
        self._dimension: int | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    @property
    def dimension(self) -> int:
        # bge-small-en-v1.5 -> 384. Computed once (cached) to stay model-agnostic.
        if self._dimension is None:
            self._dimension = len(self.embed_query("dimension probe"))
        return self._dimension
