"""
Embedding Provider — Real semantic vectors for the Creativity Engine.

Replaces the heuristic-based "semantic distance" proxy with actual
cosine distances in embedding space. This is what lets the engine
measure how wild a conceptual leap really is — the difference between
"coffee → caffeine" (tiny distance) and "silver coins → cancer cure"
(massive distance, exactly what we want to reward).

Supports two backends:
  1. OpenAI text-embedding-3-small (API, cheap, 1536-dim)
  2. sentence-transformers all-MiniLM-L6-v2 (local, free, 384-dim)

The engine uses whichever is available, preferring local for speed.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class EmbeddingConfig:
    provider: str = "openai"        # "openai" or "local"
    openai_model: str = "text-embedding-3-small"
    local_model: str = "all-MiniLM-L6-v2"
    api_key_env: str = "OPENAI_API_KEY"
    cache_enabled: bool = True      # cache embeddings to avoid redundant calls
    batch_size: int = 32            # max texts per batch embed call


class EmbeddingProvider:
    """Computes semantic embeddings for concept topics.

    Caches results so the same topic is never embedded twice within a session.
    Supports both batched and single-text embedding.
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        self.cfg = config or EmbeddingConfig()
        self._cache: dict[str, np.ndarray] = {}
        self._local_model = None
        self._openai_client = None
        self._available = False
        self._provider_name = ""
        self._dimension = 0

    def initialize(self) -> bool:
        """Try to initialize the embedding backend. Returns True if ready."""
        if self.cfg.provider == "local":
            return self._init_local()
        elif self.cfg.provider == "openai":
            return self._init_openai()
        else:
            if self._init_local():
                return True
            return self._init_openai()

    def _init_local(self) -> bool:
        """Try to load sentence-transformers for local embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer(self.cfg.local_model)
            test = self._local_model.encode(["test"])
            self._dimension = test.shape[1]
            self._available = True
            self._provider_name = f"local ({self.cfg.local_model}, {self._dimension}d)"
            print(f"   [Embeddings] {self._provider_name} — ready")
            return True
        except ImportError:
            print("   [Embeddings] sentence-transformers not installed — trying OpenAI")
            return False
        except Exception as e:
            print(f"   [Embeddings] Local model failed: {e} — trying OpenAI")
            return False

    def _init_openai(self) -> bool:
        """Try to set up OpenAI embeddings."""
        api_key = os.environ.get(self.cfg.api_key_env, "")
        if not api_key:
            print("   [Embeddings] No API key found — embeddings disabled")
            self._available = False
            return False

        try:
            from openai import AsyncOpenAI
            self._openai_client = AsyncOpenAI(api_key=api_key)
            self._dimension = 1536 if "small" in self.cfg.openai_model else 3072
            self._available = True
            self._provider_name = f"OpenAI ({self.cfg.openai_model}, {self._dimension}d)"
            print(f"   [Embeddings] {self._provider_name} — ready")
            return True
        except ImportError:
            print("   [Embeddings] openai package not installed — embeddings disabled")
            self._available = False
            return False

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> np.ndarray | None:
        """Embed a single text. Returns the vector or None on failure."""
        if not self._available:
            return None

        cache_key = text.strip().lower()
        if self.cfg.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        result = await self.embed_batch([text])
        if result and result[0] is not None:
            return result[0]
        return None

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray | None]:
        """Embed multiple texts in one call. Returns list of vectors (or None per failure)."""
        if not self._available:
            return [None] * len(texts)

        uncached_indices = []
        uncached_texts = []
        results: list[np.ndarray | None] = [None] * len(texts)

        for i, text in enumerate(texts):
            cache_key = text.strip().lower()
            if self.cfg.cache_enabled and cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if not uncached_texts:
            return results

        try:
            if self._local_model is not None:
                vectors = await self._embed_local(uncached_texts)
            else:
                vectors = await self._embed_openai(uncached_texts)

            for idx, vec in zip(uncached_indices, vectors):
                if vec is not None:
                    results[idx] = vec
                    if self.cfg.cache_enabled:
                        cache_key = texts[idx].strip().lower()
                        self._cache[cache_key] = vec

        except Exception as e:
            print(f"   [Embeddings] Batch embed error: {e}")

        return results

    async def _embed_local(self, texts: list[str]) -> list[np.ndarray]:
        """Embed using local sentence-transformers model."""
        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(
            None, lambda: self._local_model.encode(texts, normalize_embeddings=True)
        )
        return [vectors[i] for i in range(len(texts))]

    async def _embed_openai(self, texts: list[str]) -> list[np.ndarray | None]:
        """Embed using OpenAI API with batching."""
        all_vectors: list[np.ndarray | None] = []

        for i in range(0, len(texts), self.cfg.batch_size):
            batch = texts[i:i + self.cfg.batch_size]
            try:
                response = await self._openai_client.embeddings.create(
                    model=self.cfg.openai_model,
                    input=batch,
                )
                for item in response.data:
                    vec = np.array(item.embedding, dtype=np.float32)
                    vec = vec / (np.linalg.norm(vec) + 1e-10)
                    all_vectors.append(vec)
            except Exception as e:
                print(f"   [Embeddings] OpenAI batch error: {e}")
                all_vectors.extend([None] * len(batch))

        return all_vectors

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()


# ── Math helpers used by the scorer and tree generator ──────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0.0–1.0."""
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance: 1 - similarity. 0 = identical, 1 = orthogonal."""
    return 1.0 - cosine_similarity(a, b)


def chain_semantic_spread(embeddings: list[np.ndarray]) -> float:
    """Average pairwise cosine distance across all nodes in a chain.
    Measures how much conceptual ground the chain covers overall."""
    if len(embeddings) < 2:
        return 0.0
    distances = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            distances.append(cosine_distance(embeddings[i], embeddings[j]))
    return float(np.mean(distances))
