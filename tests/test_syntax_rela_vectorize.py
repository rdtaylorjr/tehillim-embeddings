from __future__ import annotations

import numpy as np

from syntax.corpus import PhrasePsalm
from syntax.rela import SAFE_RELA_VOCABULARY
from syntax.rela_vectorize import (
    phrase_rela_1gram_psalm_vectors,
    phrase_rela_1gram_vectors,
    phrase_rela_unigram_histogram,
)

_DIM = len(SAFE_RELA_VOCABULARY)


def _psalm(*, number, phrase_rela_by_colon, nodes=None):
    nodes = nodes if nodes is not None else tuple(range(100, 100 + len(phrase_rela_by_colon)))
    return PhrasePsalm(number=number, half_verse_nodes=nodes, half_verse_rela=phrase_rela_by_colon)


class TestPhraseRelaUnigramHistogram:
    def test_sums_to_one_for_a_non_empty_colon(self):
        histogram = phrase_rela_unigram_histogram(("NA", "Appo", "Spec"))
        assert np.isclose(histogram.sum(), 1.0)

    def test_is_all_zero_for_an_empty_colon(self):
        histogram = phrase_rela_unigram_histogram(())
        assert histogram.sum() == 0.0
        assert histogram.shape == (_DIM,)

    def test_para_is_masked_into_the_na_bin_never_its_own_slot(self):
        histogram = phrase_rela_unigram_histogram(("Para", "Para", "Appo"))
        na_index = SAFE_RELA_VOCABULARY.index("NA")
        appo_index = SAFE_RELA_VOCABULARY.index("Appo")
        assert np.isclose(histogram[na_index], 2 / 3)
        assert np.isclose(histogram[appo_index], 1 / 3)


class TestPhraseRela1gramVectors:
    def test_has_dimension_of_the_safe_vocabulary(self):
        psalms = [_psalm(number=1, phrase_rela_by_colon=(("NA", "Appo"),))]
        vectors = phrase_rela_1gram_vectors(psalms)
        assert next(iter(vectors.values())).shape == (_DIM,)

    def test_keys_vectors_by_colon_node_id(self):
        psalms = [_psalm(number=1, phrase_rela_by_colon=(("NA",), ("Appo",)), nodes=(200, 201))]
        vectors = phrase_rela_1gram_vectors(psalms)
        assert set(vectors) == {200, 201}


class TestPhraseRela1gramPsalmVectors:
    def test_broadcasts_the_identical_vector_to_every_colon_node(self):
        psalms = [_psalm(number=1, phrase_rela_by_colon=(("NA",), ("Appo",)), nodes=(400, 401))]
        vectors = phrase_rela_1gram_psalm_vectors(psalms)
        assert np.allclose(vectors[400], vectors[401])

    def test_masks_para_before_pooling_across_colons(self):
        psalms = [_psalm(number=1, phrase_rela_by_colon=(("Para",), ("Appo",)), nodes=(500, 501))]
        vector = phrase_rela_1gram_psalm_vectors(psalms)[500]
        na_index = SAFE_RELA_VOCABULARY.index("NA")
        appo_index = SAFE_RELA_VOCABULARY.index("Appo")
        assert np.isclose(vector[na_index], 0.5)
        assert np.isclose(vector[appo_index], 0.5)
