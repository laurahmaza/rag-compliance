"""
Runs the labeled query set through a retriever, computes recall@k,
precision@k, MRR, and NDCG@k per query, and aggregates to a summary —
the mechanism behind "baseline denso contra híbrido, con números."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list: ...  # returns list[ScoredChunk]


@dataclass(frozen=True)
class LabeledQuery:
    id: str
    query: str
    relevant_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float


@dataclass(frozen=True)
class EvalSummary:
    retriever_name: str
    k: int
    n_queries: int
    mean_recall_at_k: float
    mean_precision_at_k: float
    mean_mrr: float
    mean_ndcg_at_k: float
    per_query: tuple[QueryResult, ...]


def evaluate_retriever(
    retriever: Retriever, queries: list[LabeledQuery], *, k: int = 10, name: str = "retriever"
) -> EvalSummary:
    results = []
    for q in queries:
        relevant = set(q.relevant_chunk_ids)
        retrieved = [sc.chunk_id for sc in retriever.search(q.query, k=k)]
        results.append(
            QueryResult(
                query_id=q.id,
                recall_at_k=recall_at_k(retrieved, relevant, k),
                precision_at_k=precision_at_k(retrieved, relevant, k),
                mrr=mrr(retrieved, relevant),
                ndcg_at_k=ndcg_at_k(retrieved, relevant, k),
            )
        )
    n = len(results)
    return EvalSummary(
        retriever_name=name,
        k=k,
        n_queries=n,
        mean_recall_at_k=sum(r.recall_at_k for r in results) / n,
        mean_precision_at_k=sum(r.precision_at_k for r in results) / n,
        mean_mrr=sum(r.mrr for r in results) / n,
        mean_ndcg_at_k=sum(r.ndcg_at_k for r in results) / n,
        per_query=tuple(results),
    )


def load_queries(path: str) -> list[LabeledQuery]:
    import yaml
    from pathlib import Path

    raw = yaml.safe_load(Path(path).read_text())
    queries = []
    seen_ids = set()
    for entry in raw["queries"]:
        q = LabeledQuery(
            id=entry["id"],
            query=entry["query"],
            relevant_chunk_ids=tuple(entry["relevant_chunk_ids"]),
        )
        if q.id in seen_ids:
            raise ValueError(f"duplicate query id: {q.id!r}")
        seen_ids.add(q.id)
        queries.append(q)
    return queries
