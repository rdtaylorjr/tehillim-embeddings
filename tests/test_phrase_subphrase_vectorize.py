from __future__ import annotations

import numpy as np

from phrase.corpus import PhrasePsalm
from phrase.subphrase import SAFE_SUBPHRASE_RELA_VOCABULARY
from phrase.subphrase_vectorize import (
    subphrase_rela_1gram_psalm_vectors,
    subphrase_rela_1gram_vectors,
    subphrase_rela_unigram_histogram,
)

_DIM = len(SAFE_SUBPHRASE_RELA_VOCABULARY)


def _psalm(*, number, rela_by_colon, nodes=None):
    nodes = nodes if nodes is not None else tuple(range(100, 100 + len(rela_by_colon)))
    return PhrasePsalm(
        number=number, half_verse_nodes=nodes, half_verse_subphrase_rela=rela_by_colon
    )


class TestSubphraseRelaUnigramHistogram:
    def test_sums_to_one_for_a_non_empty_colon(self):
        histogram = subphrase_rela_unigram_histogram(("NA", "rec", "atr"))
        assert np.isclose(histogram.sum(), 1.0)

    def test_is_all_zero_for_an_empty_colon(self):
        histogram = subphrase_rela_unigram_histogram(())
        assert histogram.sum() == 0.0
        assert histogram.shape == (_DIM,)

    def test_par_is_masked_into_the_na_bin_never_its_own_slot(self):
        histogram = subphrase_rela_unigram_histogram(("par", "par", "rec"))
        na_index = SAFE_SUBPHRASE_RELA_VOCABULARY.index("NA")
        rec_index = SAFE_SUBPHRASE_RELA_VOCABULARY.index("rec")
        assert np.isclose(histogram[na_index], 2 / 3)
        assert np.isclose(histogram[rec_index], 1 / 3)


class TestSubphraseRela1gramVectors:
    def test_has_dimension_of_the_safe_vocabulary(self):
        psalms = [_psalm(number=1, rela_by_colon=(("NA", "rec"),))]
        vectors = subphrase_rela_1gram_vectors(psalms)
        assert next(iter(vectors.values())).shape == (_DIM,)

    def test_keys_vectors_by_colon_node_id(self):
        psalms = [_psalm(number=1, rela_by_colon=(("NA",), ("rec",)), nodes=(200, 201))]
        vectors = subphrase_rela_1gram_vectors(psalms)
        assert set(vectors) == {200, 201}


class TestSubphraseRela1gramPsalmVectors:
    def test_broadcasts_the_identical_vector_to_every_colon_node(self):
        psalms = [_psalm(number=1, rela_by_colon=(("NA",), ("rec",)), nodes=(400, 401))]
        vectors = subphrase_rela_1gram_psalm_vectors(psalms)
        assert np.allclose(vectors[400], vectors[401])

    def test_masks_par_before_pooling_across_colons(self):
        psalms = [_psalm(number=1, rela_by_colon=(("par",), ("rec",)), nodes=(500, 501))]
        vector = subphrase_rela_1gram_psalm_vectors(psalms)[500]
        na_index = SAFE_SUBPHRASE_RELA_VOCABULARY.index("NA")
        rec_index = SAFE_SUBPHRASE_RELA_VOCABULARY.index("rec")
        assert np.isclose(vector[na_index], 0.5)
        assert np.isclose(vector[rec_index], 0.5)
