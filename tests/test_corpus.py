from __future__ import annotations

from pathlib import Path

import pytest

from semantic.corpus import Corpus, _strip_accents


class TestStripAccents:
    def test_removes_a_real_cantillation_mark(self):
        assert _strip_accents("֑") == ""

    def test_keeps_niqqud(self):
        assert _strip_accents("ָ") == "ָ"

    def test_keeps_plain_consonants(self):
        assert _strip_accents("שלום") == "שלום"

    def test_mixed_text_keeps_only_niqqud_and_consonants(self):
        text = "אָ֑"  # aleph, an accent, qamats
        assert _strip_accents(text) == "אָ"


class TestCorpusLoad:
    def test_raises_a_clear_error_when_bhsa_path_does_not_exist(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="BHSA"):
            Corpus.load(tmp_path / "missing")


@pytest.mark.integration
def test_extracts_all_150_psalms_with_three_text_variants_and_real_node_ids():
    corpus = Corpus.load()
    psalms = corpus.psalms()

    assert len(psalms) == 150
    assert [p.number for p in psalms] == list(range(1, 151))

    psalm_1 = next(p for p in psalms if p.number == 1)
    assert len(psalm_1.half_verses) == 14
    assert len(psalm_1.half_verses_unvocalized) == 14
    assert len(psalm_1.half_verses_niqqud_only) == 14
    assert len(psalm_1.half_verse_nodes) == 14

    # Vocalized text has niqqud. Unvocalized text (BHSA's g_cons_utf8) has none.
    assert any("ָ" in hv for hv in psalm_1.half_verses)
    assert not any("ָ" in hv for hv in psalm_1.half_verses_unvocalized)

    # Niqqud-only text keeps niqqud but drops cantillation marks present in
    # the fully vocalized text.
    assert any("ָ" in hv for hv in psalm_1.half_verses_niqqud_only)
    assert any("֑" in hv for hv in psalm_1.half_verses)
    assert not any(
        any("֑" <= ch <= "֯" for ch in hv) for hv in psalm_1.half_verses_niqqud_only
    )

    all_nodes = [node for p in psalms for node in p.half_verse_nodes]
    assert len(all_nodes) == len(set(all_nodes))
    assert all(isinstance(node, int) and node > 0 for node in all_nodes)


@pytest.mark.integration
def test_load_accepts_an_explicit_path():
    from semantic.corpus import DEFAULT_BHSA_TF_PATH

    corpus = Corpus.load(Path(DEFAULT_BHSA_TF_PATH))
    assert len(corpus.psalms()) == 150
