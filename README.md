# rag-compliance

A hybrid-retrieval RAG pipeline over regulatory text — built and evaluated
against the real NIST AI RMF 1.0, not a toy corpus.

## The question

> Given a question about AI governance obligations, can the system retrieve
> the exact framework section that answers it — and does combining lexical
> search (BM25) with semantic search (dense embeddings) actually help?

## The graph

![Recall@10 by retriever](docs/recall_comparison.png)

*(Real numbers: BM25 is deterministic and model-free; dense and hybrid use
`all-MiniLM-L6-v2` via sentence-transformers, run in Colab.)*

## The finding

Run against the real corpus (30 labeled queries, `k=10`), with a real
embedding model:

| retriever | recall@10 | precision@10 | MRR | NDCG@10 |
|---|---|---|---|---|
| BM25 | 0.900 | 0.090 | 0.717 | 0.762 |
| Dense (`all-MiniLM-L6-v2`) | 0.933 | 0.093 | 0.838 | 0.862 |
| Hybrid (RRF) | **0.967** | 0.097 | 0.821 | 0.857 |

With a real embedding model, dense retrieval beats BM25 on every metric,
and hybrid search pushes recall@10 to 0.967 — the highest of the three.
This reverses what an earlier run with a placeholder (non-semantic)
embedder suggested: there, hybrid underperformed BM25 alone (0.633 vs
0.900), because fusing a strong lexical signal with pure noise diluted it.
The lesson held in both directions — a noisy dense signal makes hybrid
*worse* than its best component; a real one makes hybrid the *best*
retriever of the three. Hybrid isn't inherently better or worse than
either single method; it inherits the quality of its weakest input.

MRR tells a complementary story: dense (0.838) and hybrid (0.821) both
beat BM25 (0.717) by a wide margin, meaning the correct chunk isn't just
present in the top 10 more often — it's closer to rank 1. For a
retrieval-augmented system, that matters beyond recall alone: a correct
chunk buried at rank 9 costs more context budget and more reranking work
than one sitting at rank 1.

## Why this matters for the precision@10 numbers

Precision@10 looks low across the board (~0.09–0.10) — this is a
k-vs-corpus-size artifact, not a retrieval failure: most queries in this
set have exactly **one** correct chunk, so precision@10 is capped at 0.10
even for a perfect retriever. Recall, MRR, and NDCG are the metrics that
actually diagnose retrieval quality here.

## Why this domain

Regulatory and legal text is exactly where retrieval quality can be
judged better than an average engineer could judge it — citing the wrong
GOVERN subcategory in a compliance answer isn't a stylistic miss, it's an
incorrect claim about what the framework requires.

## Quickstart (BM25 is real right now — no GPU, no download)

```bash
pip install -e ".[dev]"
pytest                                            # 32 tests, no model needed

# Single query against real BM25
python -m rag_compliance.cli query --text "What is residual risk?" --retriever bm25

# Full evaluation with a placeholder embedder (dense/hybrid numbers are illustrative only)
python -m rag_compliance.cli evaluate --embedder fake --k 10
```

## Running for real (Google Colab, free GPU or CPU)

```python
!git clone https://github.com/laurahmaza/rag-compliance.git
%cd rag-compliance
!pip install -e ".[model]" -q
```

```python
!python -m rag_compliance.cli evaluate --embedder all-MiniLM-L6-v2 --k 10 \
    --output results/eval_report_real.json --plot docs/recall_comparison.png
```

`all-MiniLM-L6-v2` is a small, fast sentence-transformers model — runs
fine on Colab's free CPU tier, no GPU strictly required.

## Architecture

corpus.py Chunk dataclass + YAML loader (structure-based chunking,
not fixed token count)
embeddings.py Embedder protocol: FakeEmbedder (deterministic,
dependency-free) and SentenceTransformerEmbedder (real,
lazy-imported)
index.py DenseIndex (numpy cosine similarity) and BM25Index
(rank_bm25 — real, no model needed)
hybrid.py Reciprocal rank fusion, combining two ranked lists by
rank position rather than incomparable raw scores
metrics.py recall@k, precision@k, MRR, NDCG@k — pure math,
measured independently of generation
eval.py Orchestration: run the labeled query set through a
retriever, aggregate to summary statistics
plotting.py Recall@k bar chart comparing retrievers
cli.py query and evaluate subcommands


## The corpus

`corpus/nist_ai_rmf.yaml` — 50 chunks from NIST AI 100-1 (AI RMF 1.0),
chunked by document structure: one chunk per subsection, or one chunk per
GOVERN/MAP/MEASURE/MANAGE category with its subcategories. This is a U.S.
government publication and is in the public domain (17 U.S.C. § 105) — no
licensing concern in using the full text.

## The 30 labeled queries

`cases/queries.yaml` — a deliberate mix: some queries reuse exact
framework terminology (TEVV, PETs, residual risk) where lexical search
has every advantage; others paraphrase away from the source wording
entirely, to test semantic matching.

## Limitations

- **30 queries, mostly one relevant chunk each.** This demonstrates the
  evaluation mechanism and produced real findings, but a broader claim
  about retrieval quality on regulatory text would need more queries,
  including queries with multiple correct chunks.
- **No reranking stage.** The curriculum's two-stage "retrieve broad,
  rerank fine" pattern with a cross-encoder isn't implemented here —
  retrieval is single-stage (BM25, dense, or RRF-fused).
- **BM25's strong numbers partly reflect query design.** Several queries
  reuse the framework's own section-heading vocabulary (e.g., "GOVERN 1",
  "MAP 5"), which lexical search is especially well-suited to match. A
  harder, fully paraphrased query set would likely widen dense's
  advantage over BM25 further.
- **No citation-verification layer.** This repo measures whether the
  *right chunk* is retrieved — it doesn't yet verify that a generated
  answer's claims are actually supported by the chunk it cites (that's
  Week 4 material: forcing citation by span and verifying programmatically
  that the cited text supports the claim).
- **Single embedding model tested.** `all-MiniLM-L6-v2` is small and fast;
  a larger model might shift the dense/hybrid numbers further.

## Tests

```bash
pytest -v
```

32 tests covering metrics math, corpus/query loading, both indexes, and
RRF fusion — all run against `FakeEmbedder` or fixed fake retrievers, so
no API key, GPU, or model download is required to verify the logic is
correct.