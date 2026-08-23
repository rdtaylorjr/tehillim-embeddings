from __future__ import annotations

import pytest

from lexical.surface_corpus import SurfaceCorpus, _strip_accents


class TestStripAccents:
    def test_removes_cantillation_marks_but_keeps_niqqud(self):
        # U+0591 (etnahta, cantillation) followed by U+05B4 (hiriq, niqqud).
        text = "אִ֑ב"
        assert _strip_accents(text) == "אִב"

    def test_leaves_plain_consonantal_text_unchanged(self):
        assert _strip_accents("אבג") == "אבג"


class TestSurfaceCorpusLoad:
    def test_raises_a_clear_error_when_bhsa_path_does_not_exist(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="BHSA"):
            SurfaceCorpus.load(tmp_path / "missing")


@pytest.mark.integration
def test_extracts_all_150_psalms_with_three_aligned_text_tiers():
    corpus = SurfaceCorpus.load()
    psalms = corpus.psalms()

    assert len(psalms) == 150
    assert [p.number for p in psalms] == list(range(1, 151))

    psalm_1 = next(p for p in psalms if p.number == 1)
    assert len(psalm_1.colon_nodes) == 14
    assert len(psalm_1.colon_consonantal) == 14
    assert len(psalm_1.colon_vocalized) == 14
    assert len(psalm_1.colon_cantillation) == 14

    for consonantal, vocalized, cantillation in zip(
        psalm_1.colon_consonantal,
        psalm_1.colon_vocalized,
        psalm_1.colon_cantillation,
        strict=True,
    ):
        assert len(consonantal) == len(vocalized) == len(cantillation)

    # Cantillation carries strictly more marks than vocalized (niqqud only), so a colon with
    # any cantillation-bearing word must differ, while its consonantal form is always shorter.
    first_word_cons = psalm_1.colon_consonantal[0][0]
    first_word_cant = psalm_1.colon_cantillation[0][0]
    assert len(first_word_cons) <= len(first_word_cant)
