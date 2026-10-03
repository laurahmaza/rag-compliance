"""
Public demo for rag-compliance — retrieval + citation-verifying agent
over the real NIST AI RMF 1.0.

Two tabs:
  1. Retrieve — see the raw BM25 search in action, chunk by chunk
  2. Ask the agent — ask a compliance question, get a cited answer with
     every citation checked against what was actually retrieved

Runs BM25-only by default (no model download, no API key needed) so the
public Space works instantly for anyone. If an OPENAI_API_KEY secret is
set on the Space, the agent tab uses a real LLM brain instead of the
deterministic demo brain.
"""
from __future__ import annotations

import os

import gradio as gr

from rag_compliance.agent import AgentConfig, run_agent
from rag_compliance.brain import AdaptiveFakeBrain, OpenAIBrain
from rag_compliance.corpus import load_corpus
from rag_compliance.index import BM25Index
from rag_compliance.tools import GetChunkTool, ListSectionsTool, RetrieveTool

CORPUS_PATH = "corpus/nist_ai_rmf.yaml"

_chunks = load_corpus(CORPUS_PATH)
_by_id = {c.id: c for c in _chunks}
_bm25 = BM25Index(_chunks)

_tools_schema = [
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


def do_retrieve(query: str, k: int):
    if not query.strip():
        return "Enter a question above and click Search."
    results = _bm25.search(query, k=int(k))
    if not results or all(sc.score == 0.0 for sc in results):
        return "No matching chunks found for this query — the corpus has nothing relevant to it."
    lines = []
    for sc in results:
        chunk = _by_id[sc.chunk_id]
        lines.append(f"**[{sc.score:.3f}] {chunk.id}** — {chunk.section}\n\n{chunk.text}\n")
    return "\n---\n".join(lines)


def do_agent(question: str):
    if not question.strip():
        return "Enter a question above and click Ask.", "", ""

    tools = {
        "retrieve": RetrieveTool(_bm25),
        "get_chunk": GetChunkTool(_chunks),
        "list_sections": ListSectionsTool(_chunks),
    }

    using_real_llm = bool(os.environ.get("OPENAI_API_KEY"))
    if using_real_llm:
        brain = OpenAIBrain(tools_schema=_tools_schema)
    else:
        brain = AdaptiveFakeBrain()

    result = run_agent(question, brain, tools, _chunks, AgentConfig(max_steps=6))

    status_note = (
        "*(Real LLM brain — OPENAI_API_KEY is set on this Space.)*"
        if using_real_llm
        else "*(Demo brain — no LLM API key configured on this Space. "
        "Retrieves once and cites the top hit; set OPENAI_API_KEY as a Space secret "
        "for real multi-step reasoning.)*"
    )

    answer = result.answer_text or f"*No answer — status: {result.status}*"

    if result.citation_checks:
        citation_lines = []
        for c in result.citation_checks:
            flag = "✅ grounded" if c.grounded and c.exists_in_corpus else "⚠️ UNGROUNDED"
            citation_lines.append(f"- **[{flag}]** `{c.chunk_id}`: {c.claim}")
        citations_md = "\n".join(citation_lines)
    else:
        citations_md = "*(no citations given)*"

    trace_lines = [f"`[step {s.step_number}]` **{s.kind}** — {s.detail[:200]}" for s in result.trace]
    trace_md = "\n\n".join(trace_lines)

    return f"{status_note}\n\n{answer}", citations_md, trace_md


with gr.Blocks(title="rag-compliance — NIST AI RMF") as demo:
    gr.Markdown(
        "# rag-compliance\n"
        "Hybrid-retrieval RAG + citation-verifying agent over the real "
        "**NIST AI RMF 1.0**. Every citation the agent gives is checked "
        "programmatically against what it actually retrieved — not trusted "
        "because the model said so. "
        "[Source & findings](https://github.com/laurahmaza/rag-compliance)"
    )

    with gr.Tab("Retrieve"):
        gr.Markdown("Raw BM25 search over the 50-chunk corpus — no model, no API key needed.")
        with gr.Row():
            query_box = gr.Textbox(label="Query", placeholder="What is residual risk?", scale=4)
            k_box = gr.Number(label="Top k", value=5, precision=0, scale=1)
        search_btn = gr.Button("Search", variant="primary")
        retrieve_out = gr.Markdown()
        search_btn.click(do_retrieve, inputs=[query_box, k_box], outputs=retrieve_out)
        gr.Examples(
            examples=[
                "What is residual risk?",
                "What does GOVERN 6 require about third-party software?",
                "What is the difference between explainability and interpretability?",
            ],
            inputs=query_box,
        )

    with gr.Tab("Ask the agent"):
        gr.Markdown(
            "Ask a compliance question. The agent retrieves supporting text and answers "
            "with citations — each one checked against what it actually retrieved."
        )
        question_box = gr.Textbox(label="Question", placeholder="What is residual risk?")
        ask_btn = gr.Button("Ask", variant="primary")
        answer_out = gr.Markdown(label="Answer")
        gr.Markdown("### Citations")
        citations_out = gr.Markdown()
        with gr.Accordion("Full trace (debug)", open=False):
            trace_out = gr.Markdown()
        ask_btn.click(do_agent, inputs=question_box, outputs=[answer_out, citations_out, trace_out])
        gr.Examples(
            examples=[
                "What is residual risk?",
                "What does GOVERN 6 require about third-party risk?",
                "What is the capital of France?",
            ],
            inputs=question_box,
        )

    gr.Markdown(
        "---\n"
        "Built as part of a 12-week AI Evals Engineer portfolio. "
        "[Research log](https://substack.com/@laurahmaza) · "
        "[laurahmaza.com](https://laurahmaza.com)"
    )

if __name__ == "__main__":
    # Render (and most PaaS free tiers) assign a port via $PORT and expect
    # the app to bind to 0.0.0.0, not localhost — this works equally well
    # for local testing (falls back to 7860) and for the deployed service.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
