"""Bar chart comparing retrievers on recall@k — the graph for the README."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .eval import EvalSummary


def plot_recall_comparison(summaries: list[EvalSummary], title: str, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    names = [s.retriever_name for s in summaries]
    values = [s.mean_recall_at_k for s in summaries]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, values)
    ax.set_ylabel(f"mean recall@{summaries[0].k}")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
