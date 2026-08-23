from __future__ import annotations

import numpy as np

from syntax.corpus import PhrasePsalm
from syntax.function_ngram import (
    phrase_function_1_2_3gram_psalm_vectors,
    phrase_function_1_2_3gram_vectors,
    phrase_function_1_2gram_psalm_vectors,
    phrase_function_1_2gram_vectors,
    phrase_function_1gram_psalm_vectors,
    phrase_function_1gram_vectors,
    phrase_function_bigram_histogram,
    phrase_function_trigram_histogram,
    phrase_function_unigram_histogram,
)
from syntax.vocabulary import FUNCTION_VOCABULARY

_DIM = len(FUNCTION_VOCABULARY)


def _psalm(*, number, phrase_function_by_colon, nodes=None):
    nodes = nodes if nodes is not None else tuple(range(100, 100 + len(phrase_function_by_colon)))
    return PhrasePsalm(number=number, colon_nodes=nodes, colon_function=phrase_function_by_colon)


class TestPhraseFunctionUnigramHistogram:
    def test_sums_to_one_for_a_non_empty_colon(self):
        histogram = phrase_function_unigram_histogram(("Subj", "Pred", "Cmpl"))
        assert np.isclose(histogram.sum(), 1.0)

    def test_is_all_zero_for_an_empty_colon(self):
        histogram = phrase_function_unigram_histogram(())
        assert histogram.sum() == 0.0
        assert histogram.shape == (_DIM,)

    def test_counts_repeated_functions_not_just_presence(self):
        histogram = phrase_function_unigram_histogram(("Subj", "Subj", "Pred"))
        subj_index = FUNCTION_VOCABULARY.index("Subj")
        pred_index = FUNCTION_VOCABULARY.index("Pred")
        assert np.isclose(histogram[subj_index], 2 / 3)
        assert np.isclose(histogram[pred_index], 1 / 3)

    def test_is_order_invariant(self):
        forward = phrase_function_unigram_histogram(("Subj", "Pred", "Cmpl"))
        reversed_ = phrase_function_unigram_histogram(("Cmpl", "Pred", "Subj"))
        assert np.allclose(forward, reversed_)


class TestPhraseFunctionBigramHistogram:
    def test_degenerates_to_zero_below_two_atoms(self):
        assert phrase_function_bigram_histogram(()).sum() == 0.0
        assert phrase_function_bigram_histogram(("Subj",)).sum() == 0.0

    def test_sums_to_one_for_a_colon_with_at_least_two_atoms(self):
        histogram = phrase_function_bigram_histogram(("Subj", "Pred", "Cmpl"))
        assert np.isclose(histogram.sum(), 1.0)

    def test_is_order_sensitive(self):
        forward = phrase_function_bigram_histogram(("Subj", "Pred", "Cmpl"))
        reversed_ = phrase_function_bigram_histogram(("Cmpl", "Pred", "Subj"))
        assert not np.allclose(forward, reversed_)


class TestPhraseFunctionTrigramHistogram:
    def test_degenerates_to_zero_below_three_atoms(self):
        assert phrase_function_trigram_histogram(("Subj", "Pred")).sum() == 0.0

    def test_sums_to_one_for_a_colon_with_at_least_three_atoms(self):
        histogram = phrase_function_trigram_histogram(("Subj", "Pred", "Objc", "Cmpl"))
        assert np.isclose(histogram.sum(), 1.0)

    def test_is_order_sensitive(self):
        forward = phrase_function_trigram_histogram(("Subj", "Pred", "Objc", "Cmpl"))
        reversed_ = phrase_function_trigram_histogram(("Cmpl", "Objc", "Pred", "Subj"))
        assert not np.allclose(forward, reversed_)


class TestPhraseFunctionNgramVectors:
    def test_1gram_vectors_has_dimension_of_the_vocabulary(self):
        psalms = [_psalm(number=1, phrase_function_by_colon=(("Subj", "Pred"),))]
        vectors = phrase_function_1gram_vectors(psalms)
        assert next(iter(vectors.values())).shape == (_DIM,)

    def test_1_2gram_vectors_concatenates_unigram_and_bigram(self):
        psalms = [_psalm(number=1, phrase_function_by_colon=(("Subj", "Pred"),))]
        vectors = phrase_function_1_2gram_vectors(psalms)
        assert next(iter(vectors.values())).shape == (_DIM + _DIM * _DIM,)

    def test_1_2_3gram_vectors_concatenates_all_three_orders(self):
        psalms = [_psalm(number=1, phrase_function_by_colon=(("Subj", "Pred"),))]
        vectors = phrase_function_1_2_3gram_vectors(psalms)
        assert next(iter(vectors.values())).shape == (_DIM + _DIM**2 + _DIM**3,)

    def test_keys_vectors_by_colon_node_id(self):
        psalms = [
            _psalm(number=1, phrase_function_by_colon=(("Subj",), ("Pred",)), nodes=(200, 201))
        ]
        vectors = phrase_function_1gram_vectors(psalms)
        assert set(vectors) == {200, 201}

    def test_order_by_node_reorders_atoms_within_that_colon_only(self):
        psalms = [
            _psalm(number=1, phrase_function_by_colon=(("Subj", "Pred", "Cmpl"),), nodes=(300,))
        ]
        reversed_order = {300: np.array([2, 1, 0])}

        unshuffled = phrase_function_1_2gram_vectors(psalms)[300]
        shuffled = phrase_function_1_2gram_vectors(psalms, order_by_node=reversed_order)[300]

        expected = np.concatenate(
            [
                phrase_function_unigram_histogram(("Cmpl", "Pred", "Subj")),
                phrase_function_bigram_histogram(("Cmpl", "Pred", "Subj")),
            ]
        )
        assert np.allclose(shuffled, expected)
        assert not np.allclose(unshuffled, shuffled)


class TestPhraseFunctionNgramPsalmVectors:
    def test_broadcasts_the_identical_vector_to_every_colon_node(self):
        psalms = [
            _psalm(number=1, phrase_function_by_colon=(("Subj",), ("Pred",)), nodes=(400, 401))
        ]
        vectors = phrase_function_1gram_psalm_vectors(psalms)
        assert np.allclose(vectors[400], vectors[401])

    def test_pools_raw_atom_counts_across_colons_before_normalizing_once(self):
        psalms = [
            _psalm(
                number=1,
                phrase_function_by_colon=(("Subj",), ("Pred", "Pred", "Cmpl")),
                nodes=(500, 501),
            )
        ]
        vector = phrase_function_1gram_psalm_vectors(psalms)[500]
        subj_i, pred_i, cmpl_i = (FUNCTION_VOCABULARY.index(v) for v in ("Subj", "Pred", "Cmpl"))
        assert np.isclose(vector[subj_i], 0.25)
        assert np.isclose(vector[pred_i], 0.5)
        assert np.isclose(vector[cmpl_i], 0.25)

    def test_1_2gram_psalm_vectors_has_the_cumulative_dimension(self):
        psalms = [_psalm(number=1, phrase_function_by_colon=(("Subj", "Pred"),))]
        vectors = phrase_function_1_2gram_psalm_vectors(psalms)
        assert next(iter(vectors.values())).shape == (_DIM + _DIM * _DIM,)

    def test_1_2_3gram_psalm_vectors_has_the_cumulative_dimension(self):
        psalms = [_psalm(number=1, phrase_function_by_colon=(("Subj", "Pred"),))]
        vectors = phrase_function_1_2_3gram_psalm_vectors(psalms)
        assert next(iter(vectors.values())).shape == (_DIM + _DIM**2 + _DIM**3,)

    def test_order_by_node_reorders_each_colons_atoms_before_pooling(self):
        psalms = [
            _psalm(number=1, phrase_function_by_colon=(("Subj", "Pred", "Cmpl"),), nodes=(600,))
        ]
        order = {600: np.array([2, 1, 0])}
        unshuffled = phrase_function_1_2gram_psalm_vectors(psalms)[600]
        shuffled = phrase_function_1_2gram_psalm_vectors(psalms, order_by_node=order)[600]
        assert not np.allclose(unshuffled, shuffled)
