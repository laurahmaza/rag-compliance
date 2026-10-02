from rag_compliance.agent import AgentConfig, run_agent
from rag_compliance.brain import FakeBrain, call, final, invalid
from rag_compliance.corpus import Chunk
from rag_compliance.tools import GetChunkTool, RetrieveTool
from rag_compliance.index import BM25Index


def _corpus():
    return [
        Chunk(id="c1", section="s1", text="residual risk remains after treatment is applied"),
        Chunk(id="c2", section="s2", text="transparency enables accountability in systems"),
    ]


def _tools(corpus):
    return {
        "retrieve": RetrieveTool(BM25Index(corpus)),
        "get_chunk": GetChunkTool(corpus),
    }


class TestHappyPath:
    def test_retrieve_then_answer_with_grounded_citation(self):
        corpus = _corpus()
        script = [
            call("retrieve", {"query": "residual risk"}),
            final("Residual risk is risk left after treatment.", citations=[("c1", "residual risk remains after treatment")]),
        ]
        brain = FakeBrain(script)
        result = run_agent("What is residual risk?", brain, _tools(corpus), corpus)

        assert result.status == "success"
        assert result.answer_text is not None
        assert len(result.citation_checks) == 1
        assert result.citation_checks[0].grounded


class TestFailureMode1_ToolHallucination:
    def test_unknown_tool_is_caught_not_crashed(self):
        corpus = _corpus()
        script = [
            call("nonexistent_tool", {"query": "x"}),
            call("retrieve", {"query": "residual risk"}),
            final("answer", citations=[("c1", "residual risk remains after treatment")]),
        ]
        brain = FakeBrain(script)
        result = run_agent("q", brain, _tools(corpus), corpus, AgentConfig(max_tool_errors=2))

        assert result.status == "success"  # recovered after the bad call
        error_steps = [s for s in result.trace if s.kind == "error"]
        assert len(error_steps) == 1
        assert "unknown tool" in error_steps[0].detail

    def test_too_many_tool_errors_aborts(self):
        corpus = _corpus()
        script = [call("bad_tool_1", {}), call("bad_tool_2", {}), call("bad_tool_3", {})]
        brain = FakeBrain(script)
        result = run_agent("q", brain, _tools(corpus), corpus, AgentConfig(max_tool_errors=1, max_steps=5))

        assert result.status == "too_many_tool_errors"
        assert result.answer_text is None


class TestFailureMode2_StepBudgetExceeded:
    def test_never_reaching_final_answer_stops_at_budget(self):
        corpus = _corpus()
        # always retrieve, never answer
        script = [call("retrieve", {"query": "x"}) for _ in range(10)]
        brain = FakeBrain(script)
        result = run_agent("q", brain, _tools(corpus), corpus, AgentConfig(max_steps=3))

        assert result.status == "step_budget_exceeded"
        assert len([s for s in result.trace if s.kind == "tool_call"]) == 3


class TestFailureMode3_MalformedOutput:
    def test_single_invalid_output_gets_one_retry(self):
        corpus = _corpus()
        script = [invalid(), final("recovered", citations=[("c1", "residual risk remains after treatment")])]
        brain = FakeBrain(script)
        result = run_agent("q", brain, _tools(corpus), corpus)

        assert result.status == "success"  # recovered on retry

    def test_two_consecutive_invalid_outputs_aborts(self):
        corpus = _corpus()
        script = [invalid(), invalid()]
        brain = FakeBrain(script)
        result = run_agent("q", brain, _tools(corpus), corpus)

        assert result.status == "invalid_output"
        assert result.answer_text is None


class TestFailureMode4_CitationHallucination:
    def test_citing_a_chunk_never_retrieved_is_flagged_not_hidden(self):
        corpus = _corpus()
        # agent never calls retrieve or get_chunk at all, just answers and cites c2
        script = [final("answer", citations=[("c2", "transparency enables accountability")])]
        brain = FakeBrain(script)
        result = run_agent("q", brain, _tools(corpus), corpus)

        assert result.status == "success"  # the RUN succeeds...
        assert len(result.citation_checks) == 1
        assert not result.citation_checks[0].grounded  # ...but the citation is caught as ungrounded

    def test_citing_a_chunk_that_does_not_exist_is_flagged(self):
        corpus = _corpus()
        script = [final("answer", citations=[("totally_made_up_id", "some claim")])]
        brain = FakeBrain(script)
        result = run_agent("q", brain, _tools(corpus), corpus)

        assert not result.citation_checks[0].exists_in_corpus


class TestFailureMode5_EmptyRetrieval:
    def test_no_matches_does_not_crash_and_agent_can_report_it(self):
        corpus = _corpus()
        script = [
            call("retrieve", {"query": "zzz nonexistent gibberish"}),
            final("I don't have information on this in the framework.", citations=[]),
        ]
        brain = FakeBrain(script)
        result = run_agent("some unanswerable question", brain, _tools(corpus), corpus)

        assert result.status == "success"
        tool_results = [s for s in result.trace if s.kind == "tool_result"]
        assert "No matching" in tool_results[0].detail
        assert result.citation_checks == ()  # honestly cited nothing, rather than fabricating
