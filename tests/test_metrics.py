import pytest

from rag_compliance.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k


class TestRecallAtK:
    def test_all_relevant_found(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "c"}
        assert recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_partial_recall(self):
        retrieved = ["a", "x", "y"]
        relevant = {"a", "c"}
        assert recall_at_k(retrieved, relevant, k=3) == 0.5

    def test_zero_recall(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a"}
        assert recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_k_truncates_before_computing(self):
        retrieved = ["x", "y", "a"]  # relevant item is rank 3
        relevant = {"a"}
        assert recall_at_k(retrieved, relevant, k=2) == 0.0
        assert recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_raises_on_empty_relevant_set(self):
        with pytest.raises(ValueError):
            recall_at_k(["a"], set(), k=3)


class TestPrecisionAtK:
    def test_all_hits(self):
        retrieved = ["a", "b"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, k=2) == 1.0

    def test_half_hits(self):
        retrieved = ["a", "x"]
        relevant = {"a"}
        assert precision_at_k(retrieved, relevant, k=2) == 0.5

    def test_empty_retrieved_is_zero(self):
        assert precision_at_k([], {"a"}, k=5) == 0.0


class TestMRR:
    def test_first_rank_hit(self):
        assert mrr(["a", "b"], {"a"}) == 1.0

    def test_second_rank_hit(self):
        assert mrr(["x", "a"], {"a"}) == pytest.approx(0.5)

    def test_no_hit_is_zero(self):
        assert mrr(["x", "y"], {"a"}) == 0.0


class TestNDCGAtK:
    def test_perfect_ranking_is_one(self):
        retrieved = ["a", "b"]
        relevant = {"a", "b"}
        assert ndcg_at_k(retrieved, relevant, k=2) == pytest.approx(1.0)

    def test_reversed_ranking_is_less_than_one(self):
        # only one relevant item, found at rank 2 instead of rank 1
        retrieved = ["x", "a"]
        relevant = {"a"}
        score = ndcg_at_k(retrieved, relevant, k=2)
        assert 0 < score < 1.0

    def test_no_hits_is_zero(self):
        assert ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0
