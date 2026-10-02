"""
A "brain" decides the agent's next move: call a tool, or give a final
answer. Two implementations:

  - FakeBrain: plays back a fixed, hand-written script of actions. Lets
    every failure mode in agent.py be tested deterministically — including
    ones a real model would rarely produce on demand, like hallucinating a
    tool name or returning malformed output.
  - OpenAIBrain: real, via an OpenAI-compatible chat completions API with
    function calling. Needs OPENAI_API_KEY and the `openai` package —
    meant for the Colab run, not the test suite.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

INVALID_OUTPUT = object()  # sentinel: FakeBrain returns this to simulate a parse failure


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    claim: str


@dataclass(frozen=True)
class ToolCallAction:
    tool_name: str
    tool_args: dict


@dataclass(frozen=True)
class FinalAnswerAction:
    text: str
    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class BrainResponse:
    action: ToolCallAction | FinalAnswerAction | object  # object case = INVALID_OUTPUT
    tokens_in: int
    tokens_out: int


class Brain(Protocol):
    model_name: str

    def decide(self, question: str, trace_summary: str, error_feedback: str | None) -> BrainResponse: ...


class FakeBrain:
    """
    Plays back a fixed script of BrainResponse objects, one per call,
    regardless of what's actually in the trace. Deterministic and
    dependency-free — exactly the FakeModel/FakeEmbedder pattern used
    elsewhere in this portfolio.
    """

    model_name = "fake-brain-v0"

    def __init__(self, script: list[BrainResponse]):
        self._script = list(script)
        self._call_count = 0

    def decide(self, question: str, trace_summary: str, error_feedback: str | None) -> BrainResponse:
        if self._call_count >= len(self._script):
            # script exhausted without a FinalAnswer — treat as invalid output
            # rather than raising, so the agent loop's own budget logic is
            # what terminates the run, not a Python exception.
            return BrainResponse(action=INVALID_OUTPUT, tokens_in=10, tokens_out=0)
        response = self._script[self._call_count]
        self._call_count += 1
        return response


def final(text: str, citations: list[tuple[str, str]], tokens_in: int = 50, tokens_out: int = 30) -> BrainResponse:
    """Convenience constructor for a scripted FinalAnswerAction response."""
    cites = tuple(Citation(chunk_id=cid, claim=claim) for cid, claim in citations)
    return BrainResponse(action=FinalAnswerAction(text=text, citations=cites), tokens_in=tokens_in, tokens_out=tokens_out)


def call(tool_name: str, tool_args: dict, tokens_in: int = 50, tokens_out: int = 15) -> BrainResponse:
    """Convenience constructor for a scripted ToolCallAction response."""
    return BrainResponse(
        action=ToolCallAction(tool_name=tool_name, tool_args=tool_args), tokens_in=tokens_in, tokens_out=tokens_out
    )


def invalid(tokens_in: int = 50, tokens_out: int = 5) -> BrainResponse:
    """Convenience constructor for a scripted malformed-output response."""
    return BrainResponse(action=INVALID_OUTPUT, tokens_in=tokens_in, tokens_out=tokens_out)


class AdaptiveFakeBrain:
    """
    A reactive (but still model-free) brain for the CLI demo: retrieves
    once, reads what came back from the trace, and cites whatever the top
    hit was — or reports it has nothing, on empty retrieval. Distinct from
    FakeBrain (which plays back a fixed script) because the CLI demo needs
    to react to real retrieval results against an arbitrary question, not
    rehearse a pre-written answer.
    """

    model_name = "adaptive-fake-brain-v0"

    def __init__(self):
        self._step = 0

    def decide(self, question: str, trace_summary: str, error_feedback: str | None) -> BrainResponse:
        self._step += 1
        if self._step == 1:
            return call("retrieve", {"query": question}, tokens_in=40, tokens_out=10)

        import re

        match = re.search(r"- (\S+) \(score=", trace_summary)
        if match is None:
            return final(
                "I don't have information on this in the NIST AI RMF.",
                citations=[],
                tokens_in=60,
                tokens_out=15,
            )
        chunk_id = match.group(1)
        return final(
            f"Based on the retrieved section ({chunk_id}), see that chunk for the relevant framework language.",
            citations=[(chunk_id, "supporting evidence for the answer")],
            tokens_in=60,
            tokens_out=25,
        )


class OpenAIBrain:
    """
    Real brain via an OpenAI-compatible chat completions API with function
    calling. The three tools are exposed as function schemas; the model's
    tool_calls or final message content is parsed back into an Action.
    """

    def __init__(self, tools_schema: list[dict], model: str = "gpt-4o-mini", base_url: str | None = None):
        self.model_name = model
        self._tools_schema = tools_schema
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            import os

            import openai

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set in environment")
            self._client = openai.OpenAI(api_key=api_key, base_url=self._base_url)
        return self._client

    def decide(self, question: str, trace_summary: str, error_feedback: str | None) -> BrainResponse:
        client = self._get_client()
        system = (
            "You are a compliance research assistant answering questions about the "
            "NIST AI RMF. Use the retrieve and get_chunk tools to find supporting text "
            "before answering. When you give a final answer, call the "
            "'final_answer' function with your answer text and a list of citations, "
            "each citation naming the exact chunk_id you retrieved that supports that "
            "specific claim. Never cite a chunk_id you did not actually retrieve."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question: {question}\n\nProgress so far:\n{trace_summary or '(nothing yet)'}"},
        ]
        if error_feedback:
            messages.append({"role": "user", "content": f"Your last output was invalid: {error_feedback}. Try again."})

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=self._tools_schema,
            tool_choice="auto",
        )
        choice = response.choices[0]
        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0

        if choice.message.tool_calls:
            tc = choice.message.tool_calls[0]
            if tc.function.name == "final_answer":
                try:
                    args = json.loads(tc.function.arguments)
                    citations = tuple(
                        Citation(chunk_id=c["chunk_id"], claim=c["claim"]) for c in args.get("citations", [])
                    )
                    action = FinalAnswerAction(text=args["text"], citations=citations)
                except (KeyError, json.JSONDecodeError):
                    action = INVALID_OUTPUT
            else:
                try:
                    args = json.loads(tc.function.arguments)
                    action = ToolCallAction(tool_name=tc.function.name, tool_args=args)
                except json.JSONDecodeError:
                    action = INVALID_OUTPUT
        else:
            action = INVALID_OUTPUT

        return BrainResponse(action=action, tokens_in=tokens_in, tokens_out=tokens_out)
