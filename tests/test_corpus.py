import pytest

from rag_compliance.corpus import load_corpus


class TestLoadCorpus:
    def test_loads_real_corpus(self):
        chunks = load_corpus("corpus/nist_ai_rmf.yaml")
        assert len(chunks) == 50
        ids = {c.id for c in chunks}
        assert "govern_1" in ids
        assert "appendixD" in ids

    def test_rejects_duplicate_ids(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "documents:\n"
            "  - id: dup\n    section: a\n    text: foo\n"
            "  - id: dup\n    section: b\n    text: bar\n"
        )
        with pytest.raises(ValueError, match="duplicate chunk id"):
            load_corpus(bad)

    def test_normalizes_whitespace(self, tmp_path):
        f = tmp_path / "c.yaml"
        f.write_text("documents:\n  - id: a\n    section: s\n    text: >\n      line one\n      line two\n")
        chunks = load_corpus(f)
        assert "\n" not in chunks[0].text
        assert chunks[0].text == "line one line two"
