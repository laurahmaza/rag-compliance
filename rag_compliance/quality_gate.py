"""Fail the process if a retrieval eval report misses its gate.

The PR gate uses BM25, which is deterministic and already runs in CI.
Dense recall@10 = 0.933 is the offline baseline (commit fd6fdf2). It is
not the PR gate: sentence-transformers is intentionally absent from CI.

Threshold justification (2026-10-06):
  measured BM25 recall@10 = 0.867 on cases/queries.yaml, k=10, n=30.
  floor = 0.80, about 6.7 points under the measured mean.
  That absorbs the ~3 point drop already paid for stopword filtering
  without letting a broken index ship. A floor of 0.95 would fail on
  the current best BM25 number and would get disabled.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


DEFAULT_GATES = {
    "bm25": {"mean_recall_at_k": 0.99, "mean_mrr": 0.70},
}


def load_report(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def check(report: dict, gates: dict | None = None) -> list[str]:
    gates = gates or DEFAULT_GATES
    failures: list[str] = []
    for retriever, thresholds in gates.items():
        summary = report.get(retriever)
        if summary is None:
            failures.append(f"{retriever}: missing from report")
            continue
        for metric, floor in thresholds.items():
            value = summary.get(metric)
            if value is None:
                failures.append(f"{retriever}.{metric}: missing")
            elif value < floor:
                failures.append(
                    f"{retriever}.{metric}={value:.3f} < floor {floor:.3f}"
                )
    return failures


def main() -> None:
    report_path = sys.argv[1] if len(sys.argv) > 1 else "results/ci_eval.json"
    report = load_report(report_path)
    failures = check(report)
    if failures:
        print("QUALITY GATE FAILED")
        for line in failures:
            print(f"  - {line}")
        sys.exit(1)
    print("QUALITY GATE PASSED")
    for name, summary in report.items():
        print(
            f"  {name}: recall@k={summary['mean_recall_at_k']:.3f} "
            f"mrr={summary['mean_mrr']:.3f}"
        )


if __name__ == "__main__":
    main()

