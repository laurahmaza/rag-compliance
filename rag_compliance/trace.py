"""
Persist an AgentResult's trace as JSONL — one line per step, plus a final
summary line. The "what to log so you can debug a failure three days
later" requirement: every step records what happened, how long it took,
and what it cost, not just the final answer.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .agent import AgentResult


def save_trace(result: AgentResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for step in result.trace:
            f.write(json.dumps(asdict(step)) + "\n")
        summary = {
            "kind": "summary",
            "status": result.status,
            "answer_text": result.answer_text,
            "citation_checks": [asdict(c) for c in result.citation_checks],
            "cost": asdict(result.cost),
        }
        f.write(json.dumps(summary) + "\n")
    return path
