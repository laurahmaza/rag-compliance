"""
The agent loop. Five failure modes are handled explicitly here, not left
to crash or silently misbehave — see each `# FAILURE MODE` comment:

  1. Tool hallucination      — brain requests a tool name that isn't registered
  2. Step budget exceeded    — too many steps without a final answer
  3. Malformed output        — brain's response doesn't parse into a valid action
  4. Citation hallucination  — final answer cites a chunk never actually retrieved
  5. Empty retrieval         — a tool call legitimately finds nothing
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .brain import INVALID_OUTPUT, Brain, FinalAnswerAction, ToolCallAction
from .citations import CitationCheck, verify_all
from .corpus import Chunk
from .tools import Tool

# Rough per-million-token pricing used only to render a cost estimate in the
# panel — not tied to any one vendor's actual billing; override via
# AgentConfig if you want a different rate.
DEFAULT_PRICE_PER_1K_TOKENS_IN = 0.00015
DEFAULT_PRICE_PER_1K_TOKENS_OUT = 0.0006


@dataclass(frozen=True)
class TraceStep:
    step_number: int
    kind: str  # "tool_call" | "tool_result" | "final_answer" | "error"
    detail: str
    tokens_in: int
    tokens_out: int
    latency_s: float
    timestamp: float


@dataclass(frozen=True)
class CostSummary:
    total_tokens_in: int
    total_tokens_out: int
    estimated_cost_usd: float
    total_latency_s: float


@dataclass(frozen=True)
class AgentResult:
    question: str
    status: str  # "success" | "step_budget_exceeded" | "too_many_tool_errors" | "invalid_output"
    answer_text: str | None
    citation_checks: tuple[CitationCheck, ...]
    trace: tuple[TraceStep, ...]
    cost: CostSummary


@dataclass
class AgentConfig:
    max_steps: int = 6
    max_tool_errors: int = 2
    price_per_1k_in: float = DEFAULT_PRICE_PER_1K_TOKENS_IN
    price_per_1k_out: float = DEFAULT_PRICE_PER_1K_TOKENS_OUT


def run_agent(
    question: str,
    brain: Brain,
    tools: dict[str, Tool],
    corpus: list[Chunk],
    config: AgentConfig | None = None,
) -> AgentResult:
    config = config or AgentConfig()
    trace: list[TraceStep] = []
    retrieved_chunk_ids: set[str] = set()
    tool_error_count = 0
    error_feedback: str | None = None
    step_number = 0

    while step_number < config.max_steps:
        step_number += 1
        start = time.monotonic()
        trace_summary = _summarize_trace(trace)
        response = brain.decide(question, trace_summary, error_feedback)
        latency = time.monotonic() - start
        error_feedback = None  # consumed this step, whether or not it's needed again

        action = response.action

        # FAILURE MODE 3: malformed / unparseable brain output
        if action is INVALID_OUTPUT:
            detail = "brain returned output that did not parse into a valid action"
            trace.append(_step(step_number, "error", detail, response, latency))
            if error_feedback is None and not _already_retried_invalid(trace):
                error_feedback = detail
                continue
            return _finish(question, "invalid_output", None, (), trace, config)

        if isinstance(action, ToolCallAction):
            tool = tools.get(action.tool_name)
            # FAILURE MODE 1: tool hallucination — requested a tool that doesn't exist
            if tool is None:
                tool_error_count += 1
                detail = f"requested unknown tool {action.tool_name!r}"
                trace.append(_step(step_number, "error", detail, response, latency))
                if tool_error_count > config.max_tool_errors:
                    return _finish(question, "too_many_tool_errors", None, (), trace, config)
                error_feedback = f"{detail}. Available tools: {', '.join(tools)}."
                continue

            trace.append(
                _step(step_number, "tool_call", f"{action.tool_name}({action.tool_args})", response, latency)
            )
            result = tool.call(**action.tool_args)
            # FAILURE MODE 5: empty retrieval is a legitimate, non-crashing result
            trace.append(TraceStep(step_number, "tool_result", result.summary, 0, 0, 0.0, time.time()))
            if action.tool_name == "retrieve" and result.data:
                retrieved_chunk_ids.update(d["chunk_id"] for d in result.data)
            elif action.tool_name == "get_chunk" and result.data:
                retrieved_chunk_ids.add(result.data["id"])
            continue

        if isinstance(action, FinalAnswerAction):
            trace.append(_step(step_number, "final_answer", action.text, response, latency))
            # FAILURE MODE 4: citation hallucination — checked, not trusted
            checks = tuple(verify_all(action.citations, retrieved_chunk_ids, corpus))
            return _finish(question, "success", action.text, checks, trace, config)

    # FAILURE MODE 2: step budget exceeded without reaching a final answer
    return _finish(question, "step_budget_exceeded", None, (), trace, config)


def _already_retried_invalid(trace: list[TraceStep]) -> bool:
    if len(trace) < 2:
        return False
    return trace[-1].kind == "error" and trace[-2].kind == "error"


def _step(step_number: int, kind: str, detail: str, response, latency: float) -> TraceStep:
    return TraceStep(
        step_number=step_number,
        kind=kind,
        detail=detail,
        tokens_in=response.tokens_in,
        tokens_out=response.tokens_out,
        latency_s=latency,
        timestamp=time.time(),
    )


def _finish(
    question: str,
    status: str,
    answer_text: str | None,
    checks: tuple[CitationCheck, ...],
    trace: list[TraceStep],
    config: AgentConfig,
) -> AgentResult:
    total_in = sum(s.tokens_in for s in trace)
    total_out = sum(s.tokens_out for s in trace)
    cost = CostSummary(
        total_tokens_in=total_in,
        total_tokens_out=total_out,
        estimated_cost_usd=(total_in / 1000 * config.price_per_1k_in) + (total_out / 1000 * config.price_per_1k_out),
        total_latency_s=sum(s.latency_s for s in trace),
    )
    return AgentResult(
        question=question,
        status=status,
        answer_text=answer_text,
        citation_checks=checks,
        trace=tuple(trace),
        cost=cost,
    )


def _summarize_trace(trace: list[TraceStep]) -> str:
    return "\n".join(f"[step {s.step_number}] {s.kind}: {s.detail}" for s in trace)
