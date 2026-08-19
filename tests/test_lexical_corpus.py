from __future__ import annotations

from pathlib import Path

import pytest

from lexical.corpus import Corpus


class TestCorpusLoad:
    def test_raises_a_clear_error_when_bhsa_path_does_not_exist(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="BHSA"):
            Corpus.load(tmp_path / "missing")


@pytest.mark.integration
def test_extracts_all_150_psalms_with_lex_and_lex0_aligned_to_words():
    corpus = Corpus.load()
    psalms = corpus.psalms()

    assert len(psalms) == 150
    assert [p.number for p in psalms] == list(range(1, 151))

    psalm_1 = next(p for p in psalms if p.number == 1)
    assert len(psalm_1.half_verse_nodes) == 14
    assert len(psalm_1.half_verse_lexemes) == 14
    assert len(psalm_1.half_verse_forms) == 14

    for lexemes, forms in zip(psalm_1.half_verse_lexemes, psalm_1.half_verse_forms, strict=True):
        assert len(lexemes) == len(forms)

    # Psalm 1's first half-verse contains the definite article, lex "H".
    assert "H" in psalm_1.half_verse_lexemes[0]

    all_nodes = [node for p in psalms for node in p.half_verse_nodes]
    assert len(all_nodes) == len(set(all_nodes))
    assert all(isinstance(node, int) and node > 0 for node in all_nodes)


@pytest.mark.integration
def test_load_accepts_an_explicit_path():
    from lexical.corpus import DEFAULT_BHSA_TF_PATH

    corpus = Corpus.load(Path(DEFAULT_BHSA_TF_PATH))
    assert len(corpus.psalms()) == 150


@pytest.mark.integration
def test_api_property_exposes_the_whole_corpus_not_just_psalms():
    from lexical.frequency import total_token_count

    corpus = Corpus.load()
    psalms_word_count = sum(len(lex) for p in corpus.psalms() for lex in p.half_verse_lexemes)

    whole_bible_word_count = total_token_count(corpus.api)

    assert whole_bible_word_count > psalms_word_count * 10
