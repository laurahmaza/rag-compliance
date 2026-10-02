"""
Three tools the agent can call. Each wraps something already built in
Week 3 (the corpus, the indexes) — the agent doesn't reimplement
retrieval, it orchestrates it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .corpus import Chunk
from .hybrid import reciprocal_rank_fusion
from .index import BM25Index, DenseIndex


@dataclass(frozen=True)
class ToolResult:
    data: object  # tool-specific payload, e.g. list of {chunk_id, section, score}
    summary: str  # short text representation, what the brain actually reads


class Tool(Protocol):
    name: str
    description: str

    def call(self, **kwargs) -> ToolResult: ...


class RetrieveTool:
    name = "retrieve"
    description = (
        "Search the NIST AI RMF corpus for chunks relevant to a query. "
        "Args: {query: str}. Returns up to 5 candidate chunk ids with scores."
    )

    def __init__(self, bm25: BM25Index, dense: DenseIndex | None = None, k: int = 5):
        self._bm25 = bm25
        self._dense = dense
        self._k = k

    def call(self, query: str) -> ToolResult:
        if self._dense is not None:
            dense_results = self._dense.search(query, k=self._k * 3)
            bm25_results = self._bm25.search(query, k=self._k * 3)
            results = reciprocal_rank_fusion(dense_results, bm25_results, top_n=self._k)
        else:
            results = self._bm25.search(query, k=self._k)

        if not results or all(sc.score == 0.0 for sc in results):
            return ToolResult(data=[], summary="No matching chunks found for this query.")

        data = [{"chunk_id": sc.chunk_id, "score": sc.score} for sc in results]
        summary_lines = [f"- {d['chunk_id']} (score={d['score']:.3f})" for d in data]
        return ToolResult(data=data, summary="\n".join(summary_lines))


class GetChunkTool:
    name = "get_chunk"
    description = "Fetch the full text of one chunk by id. Args: {chunk_id: str}."

    def __init__(self, chunks: list[Chunk]):
        self._by_id = {c.id: c for c in chunks}

    def call(self, chunk_id: str) -> ToolResult:
        chunk = self._by_id.get(chunk_id)
        if chunk is None:
            return ToolResult(data=None, summary=f"No chunk with id {chunk_id!r} exists.")
        return ToolResult(data={"id": chunk.id, "section": chunk.section, "text": chunk.text}, summary=chunk.text)


class ListSectionsTool:
    name = "list_sections"
    description = "List every chunk id and its section title in the corpus. No args."

    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks

    def call(self) -> ToolResult:
        data = [{"chunk_id": c.id, "section": c.section} for c in self._chunks]
        summary = "\n".join(f"- {d['chunk_id']}: {d['section']}" for d in data)
        return ToolResult(data=data, summary=summary)
