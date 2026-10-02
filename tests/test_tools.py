from rag_compliance.corpus import load_corpus
from rag_compliance.index import BM25Index
from rag_compliance.tools import GetChunkTool, ListSectionsTool, RetrieveTool


def _chunks():
    return load_corpus("corpus/nist_ai_rmf.yaml")


class TestRetrieveTool:
    def test_returns_results_for_real_query(self):
        chunks = _chunks()
        tool = RetrieveTool(BM25Index(chunks))
        result = tool.call(query="What is residual risk?")
        assert len(result.data) > 0
        assert "chunk_id" in result.data[0]

    def test_empty_result_for_nonsense_query(self):
        chunks = _chunks()
        tool = RetrieveTool(BM25Index(chunks))
        result = tool.call(query="zzz qqq nonexistent gibberish term")
        assert result.data == []
        assert "No matching" in result.summary


class TestGetChunkTool:
    def test_fetches_real_chunk(self):
        chunks = _chunks()
        tool = GetChunkTool(chunks)
        result = tool.call(chunk_id="govern_1")
        assert result.data["id"] == "govern_1"
        assert "GOVERN 1" in result.data["section"]

    def test_unknown_id_does_not_crash(self):
        chunks = _chunks()
        tool = GetChunkTool(chunks)
        result = tool.call(chunk_id="does_not_exist")
        assert result.data is None
        assert "does_not_exist" in result.summary


class TestListSectionsTool:
    def test_lists_all_chunks(self):
        chunks = _chunks()
        tool = ListSectionsTool(chunks)
        result = tool.call()
        assert len(result.data) == len(chunks)
