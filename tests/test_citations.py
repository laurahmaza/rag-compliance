from rag_compliance.brain import Citation
from rag_compliance.citations import verify_all, verify_citation
from rag_compliance.corpus import Chunk


def _corpus():
    return [
        Chunk(id="c1", section="s1", text="residual risk remains after treatment is applied"),
        Chunk(id="c2", section="s2", text="transparency enables accountability in systems"),
    ]


class TestVerifyCitation:
    def test_grounded_citation_passes(self):
        citation = Citation(chunk_id="c1", claim="residual risk remains after treatment")
        check = verify_citation(citation, retrieved_chunk_ids={"c1", "c2"}, corpus_by_id={c.id: c for c in _corpus()})
        assert check.grounded
        assert check.exists_in_corpus
        assert check.word_overlap > 0.5

    def test_ungrounded_citation_flagged(self):
        # chunk exists in the corpus, but was never actually retrieved this run
        citation = Citation(chunk_id="c2", claim="something about transparency")
        check = verify_citation(citation, retrieved_chunk_ids={"c1"}, corpus_by_id={c.id: c for c in _corpus()})
        assert not check.grounded
        assert check.exists_in_corpus

    def test_nonexistent_chunk_id_flagged(self):
        citation = Citation(chunk_id="does_not_exist", claim="anything")
        check = verify_citation(citation, retrieved_chunk_ids={"c1"}, corpus_by_id={c.id: c for c in _corpus()})
        assert not check.exists_in_corpus
        assert not check.grounded

    def test_low_overlap_claim_scores_low(self):
        citation = Citation(chunk_id="c1", claim="completely unrelated words here")
        check = verify_citation(citation, retrieved_chunk_ids={"c1"}, corpus_by_id={c.id: c for c in _corpus()})
        assert check.word_overlap < 0.5


class TestVerifyAll:
    def test_checks_every_citation(self):
        citations = (
            Citation(chunk_id="c1", claim="residual risk remains"),
            Citation(chunk_id="c2", claim="transparency"),
        )
        checks = verify_all(citations, retrieved_chunk_ids={"c1"}, corpus=_corpus())
        assert len(checks) == 2
        assert checks[0].grounded
        assert not checks[1].grounded
