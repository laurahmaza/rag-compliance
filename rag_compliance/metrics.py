"""
Retrieval metrics, measured separately from generation — per the curriculum:
"medí la recuperación por separado de la generación, o nunca vas a saber qué
está roto." Every function here takes a ranked list of retrieved chunk ids
and a set of known-relevant chunk ids; none of them touch an LLM.
"""
from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Of all relevant chunks, what fraction appear in the top k retrieved?"""
    if not relevant:
        raise ValueError("relevant set is empty — recall is undefined")
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Of the top k retrieved, what fraction are actually relevant?"""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for c in top_k if c in relevant)
    return hits / len(top_k)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal rank of the FIRST relevant chunk found. 0 if none found."""
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain. Relevance here is binary (1 if
    the chunk is in `relevant`, else 0) — rewards relevant chunks ranked
    higher, discounted logarithmically by position.
    """
    dcg = 0.0
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        rel = 1.0 if chunk_id in relevant else 0.0
        dcg += rel / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg
