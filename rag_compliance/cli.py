"""
Command-line entry point.

    python -m rag_compliance.cli query --text "What is residual risk?" --retriever bm25
    python -m rag_compliance.cli evaluate --embedder fake
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import AgentConfig, run_agent
from .brain import AdaptiveFakeBrain, OpenAIBrain
from .corpus import load_corpus
from .embeddings import FakeEmbedder, SentenceTransformerEmbedder
from .eval import evaluate_retriever, load_queries
from .hybrid import reciprocal_rank_fusion
from .index import BM25Index, DenseIndex
from .plotting import plot_recall_comparison
from .tools import GetChunkTool, ListSectionsTool, RetrieveTool
from .trace import save_trace


def _build_embedder(name: str):
    if name == "fake":
        return FakeEmbedder()
    return SentenceTransformerEmbedder(model_name=name)  # anything else = HF model id


class HybridRetriever:
    """Adapter so RRF-fused hybrid search has the same .search(query, k) interface."""

    def __init__(self, dense: DenseIndex, bm25: BM25Index, fetch_k: int = 30):
        self._dense = dense
        self._bm25 = bm25
        self._fetch_k = fetch_k

    def search(self, query: str, k: int = 10):
        dense_results = self._dense.search(query, k=self._fetch_k)
        bm25_results = self._bm25.search(query, k=self._fetch_k)
        return reciprocal_rank_fusion(dense_results, bm25_results, top_n=k)


def cmd_query(args: argparse.Namespace) -> None:
    chunks = load_corpus(args.corpus)
    by_id = {c.id: c for c in chunks}

    bm25 = BM25Index(chunks)
    dense = None
    if args.retriever in ("dense", "hybrid"):
        embedder = _build_embedder(args.embedder)
        dense = DenseIndex(chunks, embedder)

    if args.retriever == "bm25":
        results = bm25.search(args.text, k=args.k)
    elif args.retriever == "dense":
        results = dense.search(args.text, k=args.k)
    else:
        results = HybridRetriever(dense, bm25).search(args.text, k=args.k)

    print(f"\nquery: {args.text!r}  |  retriever: {args.retriever}\n")
    for sc in results:
        chunk = by_id[sc.chunk_id]
        print(f"  [{sc.score:.4f}] {chunk.id:<30} {chunk.section}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    chunks = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    bm25 = BM25Index(chunks)

    summaries = []
    bm25_summary = evaluate_retriever(bm25, queries, k=args.k, name="bm25")
    summaries.append(bm25_summary)

    if args.embedder:
        embedder = _build_embedder(args.embedder)
        dense = DenseIndex(chunks, embedder)
        summaries.append(evaluate_retriever(dense, queries, k=args.k, name="dense"))
        hybrid = HybridRetriever(dense, bm25)
        summaries.append(evaluate_retriever(hybrid, queries, k=args.k, name="hybrid"))

    print(f"\n{len(queries)} queries  |  k={args.k}\n")
    print(f"{'retriever':<10} {'recall@k':<10} {'precision@k':<12} {'MRR':<8} {'NDCG@k':<8}")
    for s in summaries:
        print(
            f"{s.retriever_name:<10} {s.mean_recall_at_k:<10.3f} "
            f"{s.mean_precision_at_k:<12.3f} {s.mean_mrr:<8.3f} {s.mean_ndcg_at_k:<8.3f}"
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            s.retriever_name: {
                "k": s.k,
                "n_queries": s.n_queries,
                "mean_recall_at_k": s.mean_recall_at_k,
                "mean_precision_at_k": s.mean_precision_at_k,
                "mean_mrr": s.mean_mrr,
                "mean_ndcg_at_k": s.mean_ndcg_at_k,
            }
            for s in summaries
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nsaved -> {out_path}")

    if args.plot:
        plot_path = plot_recall_comparison(summaries, f"Recall@{args.k} by retriever", args.plot)
        print(f"saved plot -> {plot_path}")


def _tools_schema() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "retrieve",
                "description": "Search the NIST AI RMF corpus for relevant chunks.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_chunk",
                "description": "Fetch the full text of one chunk by id.",
                "parameters": {
                    "type": "object",
                    "properties": {"chunk_id": {"type": "string"}},
                    "required": ["chunk_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_sections",
                "description": "List every chunk id and section title in the corpus.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "final_answer",
                "description": "Give the final answer with citations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "citations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "chunk_id": {"type": "string"},
                                    "claim": {"type": "string"},
                                },
                                "required": ["chunk_id", "claim"],
                            },
                        },
                    },
                    "required": ["text", "citations"],
                },
            },
        },
    ]


def cmd_agent(args: argparse.Namespace) -> None:
    chunks = load_corpus(args.corpus)
    bm25 = BM25Index(chunks)
    dense = None
    if args.embedder:
        dense = DenseIndex(chunks, _build_embedder(args.embedder))

    tools = {
        "retrieve": RetrieveTool(bm25, dense),
        "get_chunk": GetChunkTool(chunks),
        "list_sections": ListSectionsTool(chunks),
    }

    if args.brain == "fake":
        brain = AdaptiveFakeBrain()
    else:
        brain = OpenAIBrain(tools_schema=_tools_schema())

    result = run_agent(args.question, brain, tools, chunks, AgentConfig(max_steps=args.max_steps))

    print(f"\nquestion: {args.question!r}")
    print(f"status: {result.status}\n")
    if result.answer_text:
        print(f"answer: {result.answer_text}\n")
    if result.citation_checks:
        print("citations:")
        for c in result.citation_checks:
            flag = "OK" if c.grounded and c.exists_in_corpus else "UNGROUNDED"
            print(f"  [{flag}] {c.chunk_id}: {c.claim!r} (word_overlap={c.word_overlap:.2f})")
    print(
        f"\ncost: {result.cost.total_tokens_in} in / {result.cost.total_tokens_out} out tokens, "
        f"~${result.cost.estimated_cost_usd:.5f}, {result.cost.total_latency_s:.3f}s"
    )

    if args.trace_dir:
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in args.question[:40])
        path = Path(args.trace_dir) / f"{safe_name}.jsonl"
        save_trace(result, path)
        print(f"\nsaved trace -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag_compliance")
    sub = parser.add_subparsers(dest="command", required=True)

    p_query = sub.add_parser("query", help="run one query against a retriever")
    p_query.add_argument("--text", required=True)
    p_query.add_argument("--corpus", default="corpus/nist_ai_rmf.yaml")
    p_query.add_argument("--retriever", default="bm25", choices=["bm25", "dense", "hybrid"])
    p_query.add_argument("--embedder", default="fake", help="'fake', or an HF sentence-transformers model id")
    p_query.add_argument("--k", type=int, default=10)
    p_query.set_defaults(func=cmd_query)

    p_eval = sub.add_parser("evaluate", help="run the labeled query set, report recall/precision/MRR/NDCG")
    p_eval.add_argument("--corpus", default="corpus/nist_ai_rmf.yaml")
    p_eval.add_argument("--queries", default="cases/queries.yaml")
    p_eval.add_argument("--embedder", default=None, help="omit to run BM25 only; 'fake' or an HF model id to add dense+hybrid")
    p_eval.add_argument("--k", type=int, default=10)
    p_eval.add_argument("--output", default=None)
    p_eval.add_argument("--plot", default=None, help="path to save a recall@k bar chart PNG")
    p_eval.set_defaults(func=cmd_evaluate)

    p_agent = sub.add_parser("agent", help="run the citation-verifying agent on one question")
    p_agent.add_argument("--question", required=True)
    p_agent.add_argument("--corpus", default="corpus/nist_ai_rmf.yaml")
    p_agent.add_argument("--brain", default="fake", choices=["fake", "openai"])
    p_agent.add_argument("--embedder", default=None, help="omit for BM25-only retrieval; 'fake' or an HF model id to enable hybrid")
    p_agent.add_argument("--max-steps", type=int, default=6)
    p_agent.add_argument("--trace-dir", default="results/traces")
    p_agent.set_defaults(func=cmd_agent)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
