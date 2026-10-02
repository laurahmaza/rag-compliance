import pytest

from rag_compliance.eval import LabeledQuery, evaluate_retriever, load_queries
from rag_compliance.index import ScoredChunk


class FakeRetriever:
    """Returns a fixed, pre-scripted ranking regardless of the query text."""

    def __init__(self, ranking: list[str]):
        self._ranking = ranking

    def search(self, query: str, k: int = 10):
        return [ScoredChunk(cid, 1.0) for cid in self._ranking[:k]]


class TestLoadQueries:
    def test_loads_real_query_set(self):
        queries = load_queries("cases/queries.yaml")
        assert len(queries) == 30
        ids = {q.id for q in queries}
        assert "q01" in ids
        assert "q30" in ids

    def test_each_query_has_relevant_ids(self):
        queries = load_queries("cases/queries.yaml")
        for q in queries:
            assert len(q.relevant_chunk_ids) >= 1

    def test_rejects_duplicate_ids(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "queries:\n"
            "  - id: dup\n    query: a\n    relevant_chunk_ids: [x]\n"
            "  - id: dup\n    query: b\n    relevant_chunk_ids: [y]\n"
        )
        with pytest.raises(ValueError, match="duplicate query id"):
            load_queries(bad)


class TestEvaluateRetriever:
    def test_perfect_retriever_scores_one(self):
        queries = [LabeledQuery(id="q1", query="x", relevant_chunk_ids=("a",))]
        retriever = FakeRetriever(["a", "b", "c"])
        summary = evaluate_retriever(retriever, queries, k=3)
        assert summary.mean_recall_at_k == 1.0
        assert summary.mean_mrr == 1.0

    def test_retriever_that_never_finds_anything_scores_zero(self):
        queries = [LabeledQuery(id="q1", query="x", relevant_chunk_ids=("a",))]
        retriever = FakeRetriever(["x", "y", "z"])
        summary = evaluate_retriever(retriever, queries, k=3)
        assert summary.mean_recall_at_k == 0.0
        assert summary.mean_mrr == 0.0

    def test_averages_across_multiple_queries(self):
        queries = [
            LabeledQuery(id="q1", query="x", relevant_chunk_ids=("a",)),
            LabeledQuery(id="q2", query="y", relevant_chunk_ids=("z",)),  # never found
        ]
        retriever = FakeRetriever(["a", "b", "c"])
        summary = evaluate_retriever(retriever, queries, k=3)
        assert summary.mean_recall_at_k == 0.5  # 1.0 and 0.0 averaged
        assert summary.n_queries == 2
