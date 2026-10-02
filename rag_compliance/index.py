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
from rank_bm25 import BM25Plus

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
    """
    Uses BM25+ rather than classic Okapi BM25: standard Okapi's IDF term,
    log((N - n + 0.5) / (n + 0.5)), hits exactly zero whenever a query term
    appears in exactly half the corpus — not negative (which rank_bm25's
    epsilon parameter corrects for), just zero, silently dropping a real
    textual match's score to nothing. BM25+ adds a small positive floor to
    the term-frequency component specifically to avoid this; found via a
    real failing test on a 2-document fixture, not a theoretical concern.
    """

    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        self._ids = [c.id for c in chunks]
        tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Plus(tokenized)

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        scores = self._bm25.get_scores(_tokenize(query))
        top_idx = np.argsort(scores)[::-1][:k]
        return [ScoredChunk(self._ids[i], float(scores[i])) for i in top_idx]


# A minimal English stopword list. Without this, a completely off-topic
# query like "What is the capital of France?" still scores highest against
# SOME chunk purely on shared stopwords ("what", "is", "the", "of"), which
# makes BM25 return a confident-looking top hit for a question the corpus
# has nothing to do with — found by actually testing an off-topic query
# against the agent, not by anticipating it in advance.
_STOPWORDS = frozenset(
    "a an the of to in on for and or is are was were be been being this "
    "that these those it its as at by with from into over under than "
    "then so such not no nor do does did can could will would shall "
    "should may might must what which who whom how when where why".split()
)


def _tokenize(text: str) -> list[str]:
    return [tok for tok in text.lower().split() if tok not in _STOPWORDS]
