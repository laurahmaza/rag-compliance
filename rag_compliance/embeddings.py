"""
Anything that turns text into a fixed-length vector implements this.

  - FakeEmbedder: deterministic, dependency-free (hash-derived pseudo-random
    vectors). Same text -> same vector always; different text -> unrelated
    vectors, with NO semantic structure. Proves the retrieval pipeline runs
    end to end, but dense recall@k against it will be close to chance —
    that's expected and is called out in the README, exactly like
    llm-probe's FakeModel.
  - SentenceTransformerEmbedder: real embeddings via sentence-transformers.
    Needs the package installed and, for the first call, a model download —
    meant for Colab, not this dev loop.
"""
from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class FakeEmbedder:
    model_name = "fake-v0"

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, text in enumerate(texts):
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            v = rng.normal(size=self.dim)
            vectors[i] = v / np.linalg.norm(v)
        return vectors


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.dim = None

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        self._load()
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float64)
