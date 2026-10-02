"""
Corpus: a set of chunked documents, loaded from YAML.

Chunking strategy: by document structure (one chunk per subsection or per
GOVERN/MAP/MEASURE/MANAGE category), not fixed token count. The curriculum
is explicit that chunking strategy sets the ceiling on retrieval quality —
splitting NIST's category tables mid-list, for instance, would scatter a
GOVERN category's subcategories across chunks and make a query about any
one of them harder to retrieve cleanly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Chunk:
    id: str
    section: str
    text: str


def load_corpus(path: str | Path) -> list[Chunk]:
    raw = yaml.safe_load(Path(path).read_text())
    chunks = []
    seen_ids = set()
    for entry in raw["documents"]:
        chunk = Chunk(
            id=entry["id"],
            section=entry["section"],
            text=" ".join(entry["text"].split()),  # normalize whitespace from YAML block scalar
        )
        if chunk.id in seen_ids:
            raise ValueError(f"duplicate chunk id: {chunk.id!r}")
        seen_ids.add(chunk.id)
        chunks.append(chunk)
    return chunks
