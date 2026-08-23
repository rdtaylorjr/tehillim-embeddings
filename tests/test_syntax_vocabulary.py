from __future__ import annotations

import pytest

from syntax.vocabulary import DET_VOCABULARY, FUNCTION_VOCABULARY, TYP_VOCABULARY


class TestVocabularyContents:
    def test_typ_has_thirteen_values_no_na(self):
        assert len(TYP_VOCABULARY) == 13
        assert "NA" not in TYP_VOCABULARY

    def test_det_has_na_plus_two_real_values(self):
        assert DET_VOCABULARY == ("NA", "det", "und")

    def test_function_has_twenty_nine_values(self):
        assert len(FUNCTION_VOCABULARY) == 29

    def test_vocabularies_are_sorted_and_deduplicated(self):
        for vocabulary in (TYP_VOCABULARY, DET_VOCABULARY, FUNCTION_VOCABULARY):
            assert len(vocabulary) == len(set(vocabulary))


def _load_full_api():
    from tf.fabric import Fabric

    from syntax.corpus import DEFAULT_BHSA_TF_PATH

    tf = Fabric(locations=[str(DEFAULT_BHSA_TF_PATH)], silent="deep")
    return tf.load("otype typ det function", silent="deep")


@pytest.mark.integration
def test_typ_matches_every_value_observed_across_the_whole_bhsa_corpus():
    api = _load_full_api()
    observed = {api.F.typ.v(pa) for pa in api.F.otype.s("phrase_atom")}

    assert observed == set(TYP_VOCABULARY)


@pytest.mark.integration
def test_det_matches_every_value_observed_across_the_whole_bhsa_corpus():
    api = _load_full_api()
    observed = {api.F.det.v(pa) for pa in api.F.otype.s("phrase_atom")}

    assert observed == set(DET_VOCABULARY)


@pytest.mark.integration
def test_function_matches_every_value_observed_across_the_whole_bhsa_corpus():
    api = _load_full_api()
    observed = {api.F.function.v(p) for p in api.F.otype.s("phrase")}

    assert observed == set(FUNCTION_VOCABULARY)
