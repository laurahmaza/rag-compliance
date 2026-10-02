# rag-compliance

A hybrid-retrieval RAG pipeline over regulatory text, plus a citation-
verifying agent — built and evaluated against the real NIST AI RMF 1.0,
not a toy corpus.

## The question

> Given a question about AI governance obligations, can the system
> retrieve the exact framework section that answers it, and can an agent
> answer with citations that are actually checked against what it
> retrieved — not just trusted?

## The graph

![Recall@10 by retriever](docs/recall_comparison.png)

*(BM25 numbers below are real, current, and reflect a bug fix described
under Limitations. Dense/hybrid numbers need a fresh Colab run against the
fixed BM25 — see "Running for real.")*

## The retrieval finding (Week 3)

Run against the real corpus (30 labeled queries, `k=10`):

| retriever | recall@10 | precision@10 | MRR | NDCG@10 |
|---|---|---|---|---|
| BM25 | 0.867 | 0.087 | 0.766 | 0.791 |
| Dense (placeholder embedder) | 0.200 | 0.020 | 0.084 | 0.112 |
| Hybrid (RRF, placeholder dense) | 0.633 | 0.063 | 0.242 | 0.336 |

BM25 alone retrieves the correct section 87% of the time at k=10, with
correct chunks ranking close to the top (MRR 0.766). Recall dropped
slightly from an earlier 0.900 after adding stopword filtering (see
Limitations) — a small, deliberate tradeoff: a few queries had been
benefiting from incidental stopword overlap, and removing that made
retrieval more honest at the cost of a few points of recall. A real
embedding model (`all-MiniLM-L6-v2`) previously pushed dense to 0.933 and
hybrid to 0.967 recall@10 — but that run predates both the BM25Plus fix
and the stopword fix (see Limitations) and needs to be reproduced against
current code before those numbers can be cited again. That re-run is the
immediate next step, not yet done as of this commit.

## Why this matters for the precision@10 numbers

Precision@10 looks low across the board — this is a k-vs-corpus-size
artifact: most queries have exactly **one** correct chunk, so precision@10
is capped at 0.10 even for a perfect retriever. Recall, MRR, and NDCG are
the metrics that actually diagnose retrieval quality here.

## The agent (Week 4)

A tool-using agent that answers a compliance question by retrieving NIST
AI RMF sections, then produces an answer with **verifiable citations**:
every claim must point at a chunk id the agent actually retrieved during
that run, checked programmatically — not trusted because the model said so.

```bash
python -m rag_compliance.cli agent --question "What is residual risk?" --brain fake
```

### Tools available to the agent
- `retrieve(query)` — hybrid/BM25 search over the corpus (reuses `index.py`/`hybrid.py`)
- `get_chunk(chunk_id)` — full text of one specific chunk
- `list_sections()` — every chunk id + section title, for when the agent needs to orient itself rather than guess an id

### Five ways the agent can fail, and what the code does about each

1. **Tool hallucination** — the brain requests a tool name that isn't
   registered. Caught, logged as an error step in the trace, does not
   crash; the agent gets another chance, up to `max_tool_errors` (then
   aborts with `too_many_tool_errors`).
2. **Step budget exceeded** — the agent takes too many steps without
   reaching a final answer. The loop stops at `max_steps` and returns
   `step_budget_exceeded` with whatever partial trace exists, rather than
   looping forever.
3. **Malformed structured output** — the brain's response doesn't parse
   into a valid action. The parser error is fed back as a synthetic
   message for one retry; a second consecutive failure aborts with
   `invalid_output`.
4. **Citation hallucination** — the final answer cites a chunk id that was
   never returned by any tool call during that run. `citations.py` checks
   every citation against the trace's actually-retrieved ids and flags
   ungrounded ones explicitly in the result.
5. **Empty retrieval** — a query matches nothing in the corpus. The
   `retrieve` tool returns an explicit empty result rather than raising,
   so the agent can report "I don't have information on this" instead of
   fabricating an answer. Real example, found by actually testing an
   off-topic question rather than assuming the happy path:

   ```
   $ python -m rag_compliance.cli agent --question "What is the capital of France?" --brain fake
   answer: I don't have information on this in the NIST AI RMF.
   ```

   This only works correctly because of a second bug fix below — without
   it, the agent confidently "answered" this question by citing a random
   chunk.

All five are covered by dedicated tests in `tests/test_agent.py`, run
against `FakeBrain` — a scripted, deterministic brain that can be told to
produce exactly these failures on demand, which a real model rarely does
reliably when you actually want to test for it.

### Cost and traces

Every agent run logs a per-step token count and dollar estimate
(`agent.py`'s `CostSummary`), and persists a full JSONL trace — one line
per step — to `results/traces/`, so a failure can be debugged after the
fact instead of only at the moment it happened.

## Quickstart (BM25 and the agent's FakeBrain path are real right now)

```bash
pip install -e ".[dev]"
pytest                                            # 52 tests, no model needed

# Retrieval
python -m rag_compliance.cli query --text "What is residual risk?" --retriever bm25
python -m rag_compliance.cli evaluate --embedder fake --k 10

# Agent (deterministic fake brain — proves the loop, not model quality)
python -m rag_compliance.cli agent --question "What does GOVERN 6 require?" --brain fake
```

## Running for real (Google Colab)

Retrieval with a real embedding model:

```python
!git clone https://github.com/laurahmaza/rag-compliance.git
%cd rag-compliance
!pip install -e ".[model]" -q
!python -m rag_compliance.cli evaluate --embedder all-MiniLM-L6-v2 --k 10 \
    --output results/eval_report_real.json --plot docs/recall_comparison.png
```

The agent with a real LLM:

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
!pip install -e ".[openai]" -q
!python -m rag_compliance.cli agent --question "What does GOVERN 6 require about third-party risk?" --brain openai
```

## Architecture

```
corpus.py        Chunk dataclass + YAML loader (structure-based chunking)
embeddings.py     Embedder protocol: FakeEmbedder (deterministic) and
                  SentenceTransformerEmbedder (real, lazy-imported)
index.py          DenseIndex (numpy cosine similarity) and BM25Index
                  (rank_bm25's BM25Plus — see Limitations for why Plus,
                  not Okapi)
hybrid.py         Reciprocal rank fusion
metrics.py        recall@k, precision@k, MRR, NDCG@k
eval.py           Retrieval evaluation orchestration
plotting.py       Recall@k bar chart
tools.py          Agent tools: retrieve, get_chunk, list_sections
brain.py          Brain protocol: FakeBrain (scripted, for failure-mode
                  tests), AdaptiveFakeBrain (reactive, for the CLI demo),
                  OpenAIBrain (real, function-calling, lazy-imported)
citations.py      Programmatic citation verification — grounding, not truth
agent.py          The agent loop and its five explicit failure handlers
trace.py          JSONL trace persistence
cli.py            `query`, `evaluate`, and `agent` subcommands
```

## The corpus

`corpus/nist_ai_rmf.yaml` — 50 chunks from NIST AI 100-1 (AI RMF 1.0),
chunked by document structure. U.S. government publication, public domain
(17 U.S.C. § 105).

## The 30 labeled retrieval queries

`cases/queries.yaml` — a mix of exact-terminology and paraphrased queries
against the corpus, for recall@k/MRR evaluation.

## Limitations

- **Bug found and fixed: `BM25Okapi` → `BM25Plus`.** Classic Okapi BM25's
  IDF term is exactly zero whenever a query term appears in precisely half
  the corpus — not negative (which `rank_bm25`'s `epsilon` parameter
  corrects), just silently zero, dropping a real textual match's score to
  nothing. Found via a failing agent test on a 2-document fixture, not a
  theoretical concern — `BM25Plus` avoids it by construction. BM25 recall
  stayed flat and MRR improved (0.717 → 0.755) after this specific fix,
  before the stopword fix below was applied on top of it.
- **Bug found and fixed: no stopword filtering in BM25.** Without it, a
  completely off-topic query like "What is the capital of France?" still
  scored a confident-looking top hit against some chunk — purely from
  sharing words like "what", "is", "the". The agent would then cite that
  chunk as if it answered the question. Found by actually running the
  agent on an off-topic question, not by anticipating the edge case.
  Adding a minimal stopword list fixed it (the off-topic query now
  correctly returns zero matches) at a measured cost of ~3 points of BM25
  recall@10 on the 30-query set — a few queries had been benefiting from
  incidental stopword overlap with their correct chunk.
- **Dense/hybrid retrieval numbers need re-validation.** The 0.933/0.967
  recall figures from an earlier Colab run used the old `BM25Okapi` and
  pre-stopword-filtering tokenizer for the BM25 half of hybrid fusion —
  they need to be reproduced against current code before being cited
  again.
- **Citation verification checks grounding, not truth.** `citations.py`
  confirms a cited chunk id was actually retrieved during the run — it
  does not verify the claim is semantically entailed by the chunk's text
  (that needs an NLI-style check, not implemented here).
- **`FakeBrain` scripts are hand-written per test scenario**, not a
  general-purpose simulated LLM. They prove the agent loop's control flow
  is correct, not that a real model would behave this way — that needs
  the `--brain openai` run.
- **No reranking stage.** Retrieval is single-stage (BM25, dense, or
  RRF-fused) — no cross-encoder reranker.
- **30 retrieval queries, mostly one relevant chunk each**; a broader
  claim about retrieval quality would need more queries, including
  multi-answer ones.

## Tests

```bash
pytest -v
```

52 tests: retrieval metrics, corpus/query loading, both indexes, RRF
fusion, agent tools, citation verification, trace persistence, and — the
core of Week 4 — all five agent failure modes, each proven with a
dedicated test against `FakeBrain`. No API key, GPU, or model download
required.
