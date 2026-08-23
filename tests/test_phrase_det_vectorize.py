from __future__ import annotations

import numpy as np

from phrase.corpus import PhrasePsalm
from phrase.det_vectorize import (
    phrase_det_1gram_psalm_vectors,
    phrase_det_1gram_vectors,
    phrase_det_unigram_histogram,
)
from phrase.vocabulary import DET_VOCABULARY

_DIM = len(DET_VOCABULARY)


def _psalm(*, number, det_by_colon, nodes=None):
    nodes = nodes if nodes is not None else tuple(range(100, 100 + len(det_by_colon)))
    return PhrasePsalm(number=number, half_verse_nodes=nodes, half_verse_det=det_by_colon)


class TestPhraseDetUnigramHistogram:
    def test_sums_to_one_for_a_non_empty_colon(self):
        histogram = phrase_det_unigram_histogram(("det", "und", "NA"))
        assert np.isclose(histogram.sum(), 1.0)

    def test_is_all_zero_for_an_empty_colon(self):
        histogram = phrase_det_unigram_histogram(())
        assert histogram.sum() == 0.0
        assert histogram.shape == (_DIM,)

    def test_na_is_counted_as_part_of_the_distribution(self):
        histogram = phrase_det_unigram_histogram(("NA", "NA", "det"))
        na_index = DET_VOCABULARY.index("NA")
        det_index = DET_VOCABULARY.index("det")
        assert np.isclose(histogram[na_index], 2 / 3)
        assert np.isclose(histogram[det_index], 1 / 3)


class TestPhraseDet1gramVectors:
    def test_has_dimension_of_the_vocabulary(self):
        psalms = [_psalm(number=1, det_by_colon=(("det", "und"),))]
        vectors = phrase_det_1gram_vectors(psalms)
        assert next(iter(vectors.values())).shape == (_DIM,)

    def test_keys_vectors_by_colon_node_id(self):
        psalms = [_psalm(number=1, det_by_colon=(("det",), ("und",)), nodes=(200, 201))]
        vectors = phrase_det_1gram_vectors(psalms)
        assert set(vectors) == {200, 201}


class TestPhraseDet1gramPsalmVectors:
    def test_broadcasts_the_identical_vector_to_every_colon_node(self):
        psalms = [_psalm(number=1, det_by_colon=(("det",), ("und",)), nodes=(400, 401))]
        vectors = phrase_det_1gram_psalm_vectors(psalms)
        assert np.allclose(vectors[400], vectors[401])

    def test_pools_raw_atom_counts_across_colons_before_normalizing_once(self):
        psalms = [_psalm(number=1, det_by_colon=(("det",), ("und", "und", "NA")), nodes=(500, 501))]
        vector = phrase_det_1gram_psalm_vectors(psalms)[500]
        det_i, und_i, na_i = (DET_VOCABULARY.index(v) for v in ("det", "und", "NA"))
        assert np.isclose(vector[det_i], 0.25)
        assert np.isclose(vector[und_i], 0.5)
        assert np.isclose(vector[na_i], 0.25)
