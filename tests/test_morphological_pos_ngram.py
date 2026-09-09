from __future__ import annotations

import numpy as np

from morphological.corpus import MorphologicalPsalm
from morphological.pos_ngram import (
    pos_bigram_histogram,
    pos_trigram_histogram,
    pos_unigram_histogram,
    sp_1_2_3gram_psalm_vectors,
    sp_1_2_3gram_vectors,
    sp_1_2gram_psalm_vectors,
    sp_1_2gram_vectors,
    sp_unigram_psalm_vectors,
    sp_unigram_vectors,
)
from morphological.vocabulary import SP_VOCABULARY

_DIM = len(SP_VOCABULARY)


def _psalm(*, number, sp_by_half_verse, nodes=None):
    nodes = nodes if nodes is not None else tuple(range(100, 100 + len(sp_by_half_verse)))
    return MorphologicalPsalm(number=number, half_verse_nodes=nodes, half_verse_sp=sp_by_half_verse)


class TestPosUnigramHistogram:
    def test_sums_to_one_for_a_non_empty_half_verse(self):
        histogram = pos_unigram_histogram(("subs", "verb", "prep"))
        assert np.isclose(histogram.sum(), 1.0)

    def test_is_all_zero_for_an_empty_half_verse(self):
        histogram = pos_unigram_histogram(())
        assert histogram.sum() == 0.0
        assert histogram.shape == (_DIM,)

    def test_counts_repeated_tags_not_just_presence(self):
        histogram = pos_unigram_histogram(("subs", "subs", "verb"))
        subs_index = SP_VOCABULARY.index("subs")
        verb_index = SP_VOCABULARY.index("verb")
        assert np.isclose(histogram[subs_index], 2 / 3)
        assert np.isclose(histogram[verb_index], 1 / 3)

    def test_is_order_invariant(self):
        forward = pos_unigram_histogram(("subs", "verb", "prep"))
        reversed_ = pos_unigram_histogram(("prep", "verb", "subs"))
        assert np.allclose(forward, reversed_)


class TestPosBigramHistogram:
    def test_degenerates_to_zero_below_two_words(self):
        assert pos_bigram_histogram(()).sum() == 0.0
        assert pos_bigram_histogram(("subs",)).sum() == 0.0

    def test_sums_to_one_for_a_half_verse_with_at_least_two_words(self):
        histogram = pos_bigram_histogram(("subs", "verb", "prep"))
        assert np.isclose(histogram.sum(), 1.0)

    def test_is_order_sensitive(self):
        forward = pos_bigram_histogram(("subs", "verb", "prep"))
        reversed_ = pos_bigram_histogram(("prep", "verb", "subs"))
        assert not np.allclose(forward, reversed_)

    def test_matches_a_hand_computed_bigram_distribution(self):
        histogram = pos_bigram_histogram(("subs", "verb", "subs"))
        n = _DIM
        subs, verb = SP_VOCABULARY.index("subs"), SP_VOCABULARY.index("verb")
        assert np.isclose(histogram[subs * n + verb], 0.5)
        assert np.isclose(histogram[verb * n + subs], 0.5)


class TestPosTrigramHistogram:
    def test_degenerates_to_zero_below_three_words(self):
        assert pos_trigram_histogram(("subs", "verb")).sum() == 0.0

    def test_sums_to_one_for_a_half_verse_with_at_least_three_words(self):
        histogram = pos_trigram_histogram(("subs", "verb", "prep", "conj"))
        assert np.isclose(histogram.sum(), 1.0)

    def test_is_order_sensitive(self):
        forward = pos_trigram_histogram(("subs", "verb", "prep", "conj"))
        reversed_ = pos_trigram_histogram(("conj", "prep", "verb", "subs"))
        assert not np.allclose(forward, reversed_)


class TestSpNgramVectors:
    def test_sp_unigram_vectors_has_dimension_of_the_vocabulary(self):
        psalms = [_psalm(number=1, sp_by_half_verse=(("subs", "verb"),))]
        vectors = sp_unigram_vectors(psalms)
        assert next(iter(vectors.values())).shape == (_DIM,)

    def test_sp_1_2gram_vectors_concatenates_unigram_and_bigram(self):
        psalms = [_psalm(number=1, sp_by_half_verse=(("subs", "verb"),))]
        vectors = sp_1_2gram_vectors(psalms)
        assert next(iter(vectors.values())).shape == (_DIM + _DIM * _DIM,)

    def test_sp_1_2_3gram_vectors_concatenates_all_three_orders(self):
        psalms = [_psalm(number=1, sp_by_half_verse=(("subs", "verb"),))]
        vectors = sp_1_2_3gram_vectors(psalms)
        assert next(iter(vectors.values())).shape == (_DIM + _DIM**2 + _DIM**3,)

    def test_keys_vectors_by_half_verse_node_id(self):
        psalms = [_psalm(number=1, sp_by_half_verse=(("subs",), ("verb",)), nodes=(200, 201))]
        vectors = sp_unigram_vectors(psalms)
        assert set(vectors) == {200, 201}

    def test_order_by_node_reorders_words_within_that_half_verse_only(self):
        psalms = [_psalm(number=1, sp_by_half_verse=(("subs", "verb", "prep"),), nodes=(300,))]
        reversed_order = {300: np.array([2, 1, 0])}

        unshuffled = sp_1_2gram_vectors(psalms)[300]
        shuffled = sp_1_2gram_vectors(psalms, order_by_node=reversed_order)[300]

        expected = np.concatenate(
            [
                pos_unigram_histogram(("prep", "verb", "subs")),
                pos_bigram_histogram(("prep", "verb", "subs")),
            ]
        )
        assert np.allclose(shuffled, expected)
        assert not np.allclose(unshuffled, shuffled)


class TestSpNgramPsalmVectors:
    def test_broadcasts_the_identical_vector_to_every_half_verse_node(self):
        psalms = [_psalm(number=1, sp_by_half_verse=(("subs",), ("verb",)), nodes=(400, 401))]
        vectors = sp_unigram_psalm_vectors(psalms)
        assert np.array_equal(vectors[400], vectors[401])

    def test_pools_raw_word_counts_across_half_verses_before_normalizing_once(self):
        # Word-count weighting gives subs=1/4, verb=2/4, prep=1/4, not the per-half-verse mean.
        psalms = [
            _psalm(
                number=1, sp_by_half_verse=(("subs",), ("verb", "verb", "prep")), nodes=(500, 501)
            )
        ]
        vector = sp_unigram_psalm_vectors(psalms)[500]
        subs, verb, prep = (SP_VOCABULARY.index(v) for v in ("subs", "verb", "prep"))
        assert np.isclose(vector[subs], 0.25)
        assert np.isclose(vector[verb], 0.5)
        assert np.isclose(vector[prep], 0.25)

    def test_sp_1_2gram_psalm_vectors_has_the_cumulative_dimension(self):
        psalms = [_psalm(number=1, sp_by_half_verse=(("subs", "verb"),))]
        vectors = sp_1_2gram_psalm_vectors(psalms)
        assert next(iter(vectors.values())).shape == (_DIM + _DIM * _DIM,)

    def test_sp_1_2_3gram_psalm_vectors_has_the_cumulative_dimension(self):
        psalms = [_psalm(number=1, sp_by_half_verse=(("subs", "verb"),))]
        vectors = sp_1_2_3gram_psalm_vectors(psalms)
        assert next(iter(vectors.values())).shape == (_DIM + _DIM**2 + _DIM**3,)

    def test_order_by_node_reorders_each_half_verses_words_before_pooling(self):
        psalms = [
            _psalm(number=1, sp_by_half_verse=(("subs", "verb", "prep"),), nodes=(600,)),
        ]
        order = {600: np.array([2, 1, 0])}
        unshuffled = sp_1_2gram_psalm_vectors(psalms)[600]
        shuffled = sp_1_2gram_psalm_vectors(psalms, order_by_node=order)[600]
        assert not np.allclose(unshuffled, shuffled)
