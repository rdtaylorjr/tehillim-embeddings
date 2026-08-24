from __future__ import annotations

from pathlib import Path

import pytest

from morphology.corpus import Corpus


class TestCorpusLoad:
    def test_raises_a_clear_error_when_bhsa_path_does_not_exist(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="BHSA"):
            Corpus.load(tmp_path / "missing")


@pytest.mark.integration
def test_extracts_all_150_psalms_with_all_nine_morphology_features_aligned_to_words():
    corpus = Corpus.load()
    psalms = corpus.psalms()

    assert len(psalms) == 150
    assert [p.number for p in psalms] == list(range(1, 151))

    psalm_1 = next(p for p in psalms if p.number == 1)
    assert len(psalm_1.colon_nodes) == 14
    assert len(psalm_1.colon_sp) == 14

    feature_sequences = (
        psalm_1.colon_sp,
        psalm_1.colon_gn,
        psalm_1.colon_nu,
        psalm_1.colon_ps,
        psalm_1.colon_st,
        psalm_1.colon_vs,
        psalm_1.colon_vt,
        psalm_1.colon_prs_gn,
        psalm_1.colon_prs_nu,
        psalm_1.colon_prs_ps,
    )
    for colon_sp, *rest in zip(psalm_1.colon_sp, *feature_sequences[1:], strict=True):
        for sequence in rest:
            assert len(sequence) == len(colon_sp)

    # Psalm 1 opens with a noun ("blessed"/happy man, `>CRJ`), sp=subs.
    assert "subs" in psalm_1.colon_sp[0]

    all_nodes = [node for p in psalms for node in p.colon_nodes]
    assert len(all_nodes) == len(set(all_nodes))
    assert all(isinstance(node, int) and node > 0 for node in all_nodes)


@pytest.mark.integration
def test_sp_values_match_the_verified_fourteen_value_bhsa_inventory():
    from morphology.vocabulary import SP_VOCABULARY

    corpus = Corpus.load()
    observed = {value for p in corpus.psalms() for hv in p.colon_sp for value in hv}

    assert observed == set(SP_VOCABULARY)


@pytest.mark.integration
def test_load_accepts_an_explicit_path():
    from morphology.corpus import DEFAULT_BHSA_TF_PATH

    corpus = Corpus.load(Path(DEFAULT_BHSA_TF_PATH))
    assert len(corpus.psalms()) == 150


@pytest.mark.integration
def test_api_property_exposes_the_whole_corpus_not_just_psalms():
    corpus = Corpus.load()
    psalms_word_count = sum(len(sp) for p in corpus.psalms() for sp in p.colon_sp)

    whole_bible_word_count = len(corpus.api.F.otype.s("word"))

    assert whole_bible_word_count > psalms_word_count * 10
