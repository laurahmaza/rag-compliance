"""
Hybrid search via Reciprocal Rank Fusion (RRF): combines two ranked lists
by rank position, not raw score — sidesteps the problem that BM25 scores
and cosine similarities live on completely different, incomparable scales.

RRF(chunk) = sum over each ranking of 1 / (rrf_k + rank_in_that_list)

A chunk that ranks well in EITHER list gets a boost; a chunk that ranks
well in BOTH gets the biggest boost. rrf_k=60 is the standard constant
from the original RRF paper (Cormack et al.) — it dampens the effect of
rank differences at the very top of each list.
"""
from __future__ import annotations

from .index import ScoredChunk


def reciprocal_rank_fusion(
    *rankings: list[ScoredChunk], rrf_k: int = 60, top_n: int = 10
) -> list[ScoredChunk]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, scored in enumerate(ranking, start=1):
            fused[scored.chunk_id] = fused.get(scored.chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [ScoredChunk(chunk_id, score) for chunk_id, score in ordered]
