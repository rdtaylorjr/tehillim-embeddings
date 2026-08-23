from __future__ import annotations

import pytest

from morphology.corpus import Corpus
from morphology.vocabulary import (
    GN_VOCABULARY,
    NU_VOCABULARY,
    PRS_GN_VOCABULARY,
    PRS_NU_VOCABULARY,
    PRS_PS_VOCABULARY,
    PS_VOCABULARY,
    SP_VOCABULARY,
    ST_VOCABULARY,
    VS_VOCABULARY,
    VT_VOCABULARY,
)

_FEATURE_TO_VOCABULARY = {
    "sp": SP_VOCABULARY,
    "gn": GN_VOCABULARY,
    "nu": NU_VOCABULARY,
    "ps": PS_VOCABULARY,
    "st": ST_VOCABULARY,
    "vs": VS_VOCABULARY,
    "vt": VT_VOCABULARY,
    "prs_gn": PRS_GN_VOCABULARY,
    "prs_nu": PRS_NU_VOCABULARY,
    "prs_ps": PRS_PS_VOCABULARY,
}


class TestVocabularyContents:
    def test_gn_has_na_and_unknown_plus_two_real_values(self):
        assert GN_VOCABULARY == ("NA", "f", "m", "unknown")

    def test_nu_has_na_and_unknown_plus_three_real_values(self):
        assert NU_VOCABULARY == ("NA", "du", "pl", "sg", "unknown")

    def test_ps_has_na_and_unknown_plus_three_real_values(self):
        assert PS_VOCABULARY == ("NA", "p1", "p2", "p3", "unknown")

    def test_st_has_na_but_no_unknown(self):
        assert ST_VOCABULARY == ("NA", "a", "c", "e")
        assert "unknown" not in ST_VOCABULARY

    def test_vs_has_na_and_twenty_four_real_stem_values(self):
        assert ST_VOCABULARY[0] == "NA"
        assert len(VS_VOCABULARY) == 25
        assert VS_VOCABULARY[0] == "NA"

    def test_vt_has_na_and_eight_real_values_but_no_unknown(self):
        assert VT_VOCABULARY == (
            "NA",
            "impf",
            "impv",
            "infa",
            "infc",
            "perf",
            "ptca",
            "ptcp",
            "wayq",
        )
        assert "unknown" not in VT_VOCABULARY

    def test_prs_gn_has_na_and_unknown_plus_two_real_values(self):
        assert PRS_GN_VOCABULARY == ("NA", "f", "m", "unknown")

    def test_prs_nu_has_na_but_no_unknown_or_dual(self):
        assert PRS_NU_VOCABULARY == ("NA", "pl", "sg")
        assert "unknown" not in PRS_NU_VOCABULARY
        assert "du" not in PRS_NU_VOCABULARY

    def test_prs_ps_has_na_but_no_unknown(self):
        assert PRS_PS_VOCABULARY == ("NA", "p1", "p2", "p3")
        assert "unknown" not in PRS_PS_VOCABULARY

    def test_every_vocabulary_is_sorted(self):
        for vocabulary in _FEATURE_TO_VOCABULARY.values():
            assert list(vocabulary) == sorted(vocabulary)


@pytest.mark.integration
class TestVocabulariesCoverEveryValueObservedInPsalms:
    def test_every_psalms_observed_value_is_in_its_feature_vocabulary(self):
        corpus = Corpus.load()
        psalms = corpus.psalms()
        attribute_by_feature = {
            "sp": "colon_sp",
            "gn": "colon_gn",
            "nu": "colon_nu",
            "ps": "colon_ps",
            "st": "colon_st",
            "vs": "colon_vs",
            "vt": "colon_vt",
            "prs_gn": "colon_prs_gn",
            "prs_nu": "colon_prs_nu",
            "prs_ps": "colon_prs_ps",
        }
        for feature, vocabulary in _FEATURE_TO_VOCABULARY.items():
            attribute = attribute_by_feature[feature]
            observed = {
                value for psalm in psalms for hv in getattr(psalm, attribute) for value in hv
            }
            assert observed <= set(vocabulary), f"{feature}: observed values not in vocabulary"
