import pytest

from rag_compliance.corpus import Chunk
from rag_compliance.embeddings import FakeEmbedder
from rag_compliance.hybrid import reciprocal_rank_fusion
from rag_compliance.index import BM25Index, DenseIndex, ScoredChunk


def _chunks():
    return [
        Chunk(id="c1", section="s1", text="residual risk remains after treatment"),
        Chunk(id="c2", section="s2", text="transparency enables accountability"),
        Chunk(id="c3", section="s3", text="fairness addresses harmful bias"),
    ]


class TestDenseIndex:
    def test_returns_k_results(self):
        idx = DenseIndex(_chunks(), FakeEmbedder())
        results = idx.search("some query", k=2)
        assert len(results) == 2

    def test_deterministic_for_same_query(self):
        idx = DenseIndex(_chunks(), FakeEmbedder())
        r1 = idx.search("residual risk", k=3)
        r2 = idx.search("residual risk", k=3)
        assert [sc.chunk_id for sc in r1] == [sc.chunk_id for sc in r2]

    def test_exact_text_match_scores_highest(self):
        # querying with a chunk's own text should retrieve that chunk first
        # (cosine similarity of a vector with itself is 1.0, the maximum)
        idx = DenseIndex(_chunks(), FakeEmbedder())
        results = idx.search("residual risk remains after treatment", k=1)
        assert results[0].chunk_id == "c1"


class TestBM25Index:
    def test_returns_k_results(self):
        idx = BM25Index(_chunks())
        results = idx.search("risk", k=2)
        assert len(results) == 2

    def test_lexical_match_ranks_first(self):
        idx = BM25Index(_chunks())
        results = idx.search("residual risk treatment", k=3)
        assert results[0].chunk_id == "c1"

    def test_no_overlap_gives_zero_scores(self):
        idx = BM25Index(_chunks())
        results = idx.search("zzz qqq nonexistent", k=3)
        assert all(sc.score == 0.0 for sc in results)


class TestReciprocalRankFusion:
    def test_boosts_chunk_ranked_in_both_lists(self):
        dense = [ScoredChunk("a", 0.9), ScoredChunk("b", 0.5)]
        bm25 = [ScoredChunk("b", 10.0), ScoredChunk("a", 1.0)]
        fused = reciprocal_rank_fusion(dense, bm25, top_n=2)
        # 'a' is rank 1 in dense + rank 2 in bm25; 'b' is rank 2 in dense + rank 1 in bm25
        # both get the same two rank positions (1 and 2) summed -> tied
        fused_ids = {sc.chunk_id for sc in fused}
        assert fused_ids == {"a", "b"}

    def test_chunk_in_only_one_list_still_included(self):
        dense = [ScoredChunk("a", 0.9)]
        bm25 = [ScoredChunk("b", 10.0)]
        fused = reciprocal_rank_fusion(dense, bm25, top_n=5)
        fused_ids = {sc.chunk_id for sc in fused}
        assert fused_ids == {"a", "b"}

    def test_respects_top_n(self):
        dense = [ScoredChunk(f"c{i}", 1.0 / i) for i in range(1, 6)]
        fused = reciprocal_rank_fusion(dense, top_n=3)
        assert len(fused) == 3
