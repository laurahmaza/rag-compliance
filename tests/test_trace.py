import json

from rag_compliance.agent import AgentConfig, run_agent
from rag_compliance.brain import FakeBrain, call, final
from rag_compliance.corpus import Chunk
from rag_compliance.index import BM25Index
from rag_compliance.tools import GetChunkTool, RetrieveTool
from rag_compliance.trace import save_trace


def test_save_trace_writes_one_json_line_per_step_plus_summary(tmp_path):
    corpus = [Chunk(id="c1", section="s1", text="residual risk remains after treatment")]
    tools = {"retrieve": RetrieveTool(BM25Index(corpus)), "get_chunk": GetChunkTool(corpus)}
    script = [
        call("retrieve", {"query": "residual risk"}),
        final("answer", citations=[("c1", "residual risk remains after treatment")]),
    ]
    result = run_agent("q", FakeBrain(script), tools, corpus, AgentConfig())

    out_path = tmp_path / "trace.jsonl"
    save_trace(result, out_path)

    lines = out_path.read_text().strip().split("\n")
    # 2 tool-related steps (tool_call + tool_result) + 1 final_answer step + 1 summary line
    assert len(lines) == 4
    parsed = [json.loads(line) for line in lines]
    assert parsed[-1]["kind"] == "summary"
    assert parsed[-1]["status"] == "success"
