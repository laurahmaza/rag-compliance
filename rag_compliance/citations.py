"""
Programmatic citation verification. This is the "forzar citación por span
y verificar programáticamente que la cita respalda la afirmación" piece —
implemented at the level this harness can actually check automatically:
GROUNDING (did the agent actually retrieve this chunk during this run, or
is it citing something it never looked at?), not full semantic entailment
(whether the chunk's text truly proves the claim — that needs an NLI model,
flagged as a limitation in the README, not faked here).
"""
from __future__ import annotations

from dataclasses import dataclass

from .brain import Citation
from .corpus import Chunk


@dataclass(frozen=True)
class CitationCheck:
    chunk_id: str
    claim: str
    grounded: bool  # was this chunk_id actually retrieved during the run?
    exists_in_corpus: bool  # is this even a real chunk id?
    word_overlap: float  # crude lexical overlap between claim and chunk text


def verify_citation(citation: Citation, retrieved_chunk_ids: set[str], corpus_by_id: dict[str, Chunk]) -> CitationCheck:
    exists = citation.chunk_id in corpus_by_id
    grounded = citation.chunk_id in retrieved_chunk_ids

    overlap = 0.0
    if exists:
        chunk_words = set(corpus_by_id[citation.chunk_id].text.lower().split())
        claim_words = set(citation.claim.lower().split())
        if claim_words:
            overlap = len(chunk_words & claim_words) / len(claim_words)

    return CitationCheck(
        chunk_id=citation.chunk_id,
        claim=citation.claim,
        grounded=grounded,
        exists_in_corpus=exists,
        word_overlap=overlap,
    )


def verify_all(citations: tuple[Citation, ...], retrieved_chunk_ids: set[str], corpus: list[Chunk]) -> list[CitationCheck]:
    by_id = {c.id: c for c in corpus}
    return [verify_citation(c, retrieved_chunk_ids, by_id) for c in citations]
