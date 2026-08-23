from __future__ import annotations

import numpy as np

from phrase.corpus import PhrasePsalm
from phrase.typ_ngram import (
    phrase_typ_1_2_3gram_psalm_vectors,
    phrase_typ_1_2_3gram_vectors,
    phrase_typ_1_2gram_psalm_vectors,
    phrase_typ_1_2gram_vectors,
    phrase_typ_1gram_psalm_vectors,
    phrase_typ_1gram_vectors,
    phrase_typ_bigram_histogram,
    phrase_typ_trigram_histogram,
    phrase_typ_unigram_histogram,
)
from phrase.vocabulary import TYP_VOCABULARY

_DIM = len(TYP_VOCABULARY)


def _psalm(*, number, typ_by_colon, nodes=None):
    nodes = nodes if nodes is not None else tuple(range(100, 100 + len(typ_by_colon)))
    return PhrasePsalm(number=number, half_verse_nodes=nodes, half_verse_typ=typ_by_colon)


class TestPhraseTypUnigramHistogram:
    def test_sums_to_one_for_a_non_empty_colon(self):
        histogram = phrase_typ_unigram_histogram(("NP", "VP", "PP"))
        assert np.isclose(histogram.sum(), 1.0)

    def test_is_all_zero_for_an_empty_colon(self):
        histogram = phrase_typ_unigram_histogram(())
        assert histogram.sum() == 0.0
        assert histogram.shape == (_DIM,)

    def test_counts_repeated_types_not_just_presence(self):
        histogram = phrase_typ_unigram_histogram(("NP", "NP", "VP"))
        np_index = TYP_VOCABULARY.index("NP")
        vp_index = TYP_VOCABULARY.index("VP")
        assert np.isclose(histogram[np_index], 2 / 3)
        assert np.isclose(histogram[vp_index], 1 / 3)

    def test_is_order_invariant(self):
        forward = phrase_typ_unigram_histogram(("NP", "VP", "PP"))
        reversed_ = phrase_typ_unigram_histogram(("PP", "VP", "NP"))
        assert np.allclose(forward, reversed_)


class TestPhraseTypBigramHistogram:
    def test_degenerates_to_zero_below_two_atoms(self):
        assert phrase_typ_bigram_histogram(()).sum() == 0.0
        assert phrase_typ_bigram_histogram(("NP",)).sum() == 0.0

    def test_sums_to_one_for_a_colon_with_at_least_two_atoms(self):
        histogram = phrase_typ_bigram_histogram(("NP", "VP", "PP"))
        assert np.isclose(histogram.sum(), 1.0)

    def test_is_order_sensitive(self):
        forward = phrase_typ_bigram_histogram(("NP", "VP", "PP"))
        reversed_ = phrase_typ_bigram_histogram(("PP", "VP", "NP"))
        assert not np.allclose(forward, reversed_)


class TestPhraseTypTrigramHistogram:
    def test_degenerates_to_zero_below_three_atoms(self):
        assert phrase_typ_trigram_histogram(("NP", "VP")).sum() == 0.0

    def test_sums_to_one_for_a_colon_with_at_least_three_atoms(self):
        histogram = phrase_typ_trigram_histogram(("NP", "VP", "PP", "CP"))
        assert np.isclose(histogram.sum(), 1.0)

    def test_is_order_sensitive(self):
        forward = phrase_typ_trigram_histogram(("NP", "VP", "PP", "CP"))
        reversed_ = phrase_typ_trigram_histogram(("CP", "PP", "VP", "NP"))
        assert not np.allclose(forward, reversed_)


class TestPhraseTypNgramVectors:
    def test_1gram_vectors_has_dimension_of_the_vocabulary(self):
        psalms = [_psalm(number=1, typ_by_colon=(("NP", "VP"),))]
        vectors = phrase_typ_1gram_vectors(psalms)
        assert next(iter(vectors.values())).shape == (_DIM,)

    def test_1_2gram_vectors_concatenates_unigram_and_bigram(self):
        psalms = [_psalm(number=1, typ_by_colon=(("NP", "VP"),))]
        vectors = phrase_typ_1_2gram_vectors(psalms)
        assert next(iter(vectors.values())).shape == (_DIM + _DIM * _DIM,)

    def test_1_2_3gram_vectors_concatenates_all_three_orders(self):
        psalms = [_psalm(number=1, typ_by_colon=(("NP", "VP"),))]
        vectors = phrase_typ_1_2_3gram_vectors(psalms)
        assert next(iter(vectors.values())).shape == (_DIM + _DIM**2 + _DIM**3,)

    def test_keys_vectors_by_colon_node_id(self):
        psalms = [_psalm(number=1, typ_by_colon=(("NP",), ("VP",)), nodes=(200, 201))]
        vectors = phrase_typ_1gram_vectors(psalms)
        assert set(vectors) == {200, 201}

    def test_order_by_node_reorders_atoms_within_that_colon_only(self):
        psalms = [_psalm(number=1, typ_by_colon=(("NP", "VP", "PP"),), nodes=(300,))]
        reversed_order = {300: np.array([2, 1, 0])}

        unshuffled = phrase_typ_1_2gram_vectors(psalms)[300]
        shuffled = phrase_typ_1_2gram_vectors(psalms, order_by_node=reversed_order)[300]

        expected = np.concatenate(
            [
                phrase_typ_unigram_histogram(("PP", "VP", "NP")),
                phrase_typ_bigram_histogram(("PP", "VP", "NP")),
            ]
        )
        assert np.allclose(shuffled, expected)
        assert not np.allclose(unshuffled, shuffled)


class TestPhraseTypNgramPsalmVectors:
    def test_broadcasts_the_identical_vector_to_every_colon_node(self):
        psalms = [_psalm(number=1, typ_by_colon=(("NP",), ("VP",)), nodes=(400, 401))]
        vectors = phrase_typ_1gram_psalm_vectors(psalms)
        assert np.allclose(vectors[400], vectors[401])

    def test_pools_raw_atom_counts_across_colons_before_normalizing_once(self):
        psalms = [_psalm(number=1, typ_by_colon=(("NP",), ("VP", "VP", "PP")), nodes=(500, 501))]
        vector = phrase_typ_1gram_psalm_vectors(psalms)[500]
        np_i, vp_i, pp_i = (TYP_VOCABULARY.index(v) for v in ("NP", "VP", "PP"))
        assert np.isclose(vector[np_i], 0.25)
        assert np.isclose(vector[vp_i], 0.5)
        assert np.isclose(vector[pp_i], 0.25)

    def test_1_2gram_psalm_vectors_has_the_cumulative_dimension(self):
        psalms = [_psalm(number=1, typ_by_colon=(("NP", "VP"),))]
        vectors = phrase_typ_1_2gram_psalm_vectors(psalms)
        assert next(iter(vectors.values())).shape == (_DIM + _DIM * _DIM,)

    def test_1_2_3gram_psalm_vectors_has_the_cumulative_dimension(self):
        psalms = [_psalm(number=1, typ_by_colon=(("NP", "VP"),))]
        vectors = phrase_typ_1_2_3gram_psalm_vectors(psalms)
        assert next(iter(vectors.values())).shape == (_DIM + _DIM**2 + _DIM**3,)

    def test_order_by_node_reorders_each_colons_atoms_before_pooling(self):
        psalms = [_psalm(number=1, typ_by_colon=(("NP", "VP", "PP"),), nodes=(600,))]
        order = {600: np.array([2, 1, 0])}
        unshuffled = phrase_typ_1_2gram_psalm_vectors(psalms)[600]
        shuffled = phrase_typ_1_2gram_psalm_vectors(psalms, order_by_node=order)[600]
        assert not np.allclose(unshuffled, shuffled)
