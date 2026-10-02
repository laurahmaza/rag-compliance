"""
Two retrieval indexes, independent of each other:

  - DenseIndex: cosine similarity over embedding vectors (numpy only — no
    FAISS dependency needed at this corpus scale; a few dozen to a few
    thousand chunks is well within brute-force cosine search territory).
  - BM25Index: lexical/keyword search via rank_bm25. Needs no model, no
    embeddings, no download — runs identically everywhere, which is why
    its numbers in the README are real, not placeholders.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .corpus import Chunk
from .embeddings import Embedder


@dataclass(frozen=True)
class ScoredChunk:
    chunk_id: str
    score: float


class DenseIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder):
        self._chunks = chunks
        self._embedder = embedder
        self._ids = [c.id for c in chunks]
        self._vectors = embedder.embed([c.text for c in chunks])

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        q_vec = self._embedder.embed([query])[0]
        # vectors are normalized at embed time, so dot product == cosine similarity
        scores = self._vectors @ q_vec
        top_idx = np.argsort(scores)[::-1][:k]
        return [ScoredChunk(self._ids[i], float(scores[i])) for i in top_idx]


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        self._ids = [c.id for c in chunks]
        tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        scores = self._bm25.get_scores(_tokenize(query))
        top_idx = np.argsort(scores)[::-1][:k]
        return [ScoredChunk(self._ids[i], float(scores[i])) for i in top_idx]


def _tokenize(text: str) -> list[str]:
    return text.lower().split()
