from __future__ import annotations

import numpy as np

from lexical.corpus import LexicalPsalm
from lexical.vectorize import (
    binary_presence_vectors,
    icf_weighted_vectors,
    log_count_vectors,
    term_frequency_vectors,
    tf_icf_vectors,
)


def _psalm(*, number, lexemes, forms, nodes):
    return LexicalPsalm(
        number=number,
        half_verse_lexemes=lexemes,
        half_verse_forms=forms,
        half_verse_nodes=nodes,
    )


class TestBinaryPresenceVectors:
    def test_marks_present_lexemes_as_one_and_absent_as_zero(self):
        vocabulary = ("A", "B", "C")
        psalms = [_psalm(number=1, lexemes=(("A", "C"),), forms=((),), nodes=(100,))]

        vectors = binary_presence_vectors(psalms, vocabulary, key="lex")

        assert np.array_equal(vectors[100], [1.0, 0.0, 1.0])

    def test_a_lexeme_repeated_within_a_colon_still_scores_one_not_two(self):
        vocabulary = ("A", "B")
        psalms = [_psalm(number=1, lexemes=(("A", "A", "A"),), forms=((),), nodes=(100,))]

        vectors = binary_presence_vectors(psalms, vocabulary, key="lex")

        assert np.array_equal(vectors[100], [1.0, 0.0])

    def test_every_vector_has_length_equal_to_the_vocabulary_size(self):
        vocabulary = ("A", "B", "C", "D", "E")
        psalms = [_psalm(number=1, lexemes=(("A",), ("B", "C")), forms=((), ()), nodes=(100, 101))]

        vectors = binary_presence_vectors(psalms, vocabulary, key="lex")

        assert all(len(v) == 5 for v in vectors.values())

    def test_covers_every_half_verse_node_across_multiple_psalms(self):
        vocabulary = ("A", "B")
        psalms = [
            _psalm(number=1, lexemes=(("A",),), forms=((),), nodes=(100,)),
            _psalm(number=2, lexemes=(("B",), ("A", "B")), forms=((), ()), nodes=(200, 201)),
        ]

        vectors = binary_presence_vectors(psalms, vocabulary, key="lex")

        assert set(vectors) == {100, 200, 201}
        assert np.array_equal(vectors[201], [1.0, 1.0])

    def test_uses_lex0_forms_when_key_is_lex0(self):
        vocabulary = ("A0", "B0")
        psalms = [_psalm(number=1, lexemes=(("A", "B"),), forms=(("A0",),), nodes=(100,))]

        vectors = binary_presence_vectors(psalms, vocabulary, key="lex0")

        assert np.array_equal(vectors[100], [1.0, 0.0])

    def test_ignores_values_absent_from_the_given_vocabulary(self):
        vocabulary = ("A",)
        psalms = [_psalm(number=1, lexemes=(("A", "OUT_OF_VOCAB"),), forms=((),), nodes=(100,))]

        vectors = binary_presence_vectors(psalms, vocabulary, key="lex")

        assert np.array_equal(vectors[100], [1.0])

    def test_vectors_are_float32(self):
        vocabulary = ("A",)
        psalms = [_psalm(number=1, lexemes=(("A",),), forms=((),), nodes=(100,))]

        vectors = binary_presence_vectors(psalms, vocabulary, key="lex")

        assert vectors[100].dtype == np.float32

    def test_matches_thresholding_term_frequency_vectors_exactly(self):
        vocabulary = ("A", "B", "C")
        psalms = [
            _psalm(number=1, lexemes=(("A", "A", "C"), ("B",)), forms=((), ()), nodes=(100, 101))
        ]

        binary = binary_presence_vectors(psalms, vocabulary, key="lex")
        counts = term_frequency_vectors(psalms, vocabulary, key="lex")

        for node in binary:
            expected = (counts[node] > 0).astype(np.float32)
            assert np.array_equal(binary[node], expected)


class TestTermFrequencyVectors:
    def test_counts_repeated_occurrences_within_a_colon(self):
        vocabulary = ("A", "B")
        psalms = [_psalm(number=1, lexemes=(("A", "A", "A", "B"),), forms=((),), nodes=(100,))]

        vectors = term_frequency_vectors(psalms, vocabulary, key="lex")

        assert np.array_equal(vectors[100], [3.0, 1.0])

    def test_absent_lexeme_counts_zero(self):
        vocabulary = ("A", "B")
        psalms = [_psalm(number=1, lexemes=(("A",),), forms=((),), nodes=(100,))]

        vectors = term_frequency_vectors(psalms, vocabulary, key="lex")

        assert np.array_equal(vectors[100], [1.0, 0.0])

    def test_vectors_are_float32(self):
        vocabulary = ("A",)
        psalms = [_psalm(number=1, lexemes=(("A",),), forms=((),), nodes=(100,))]

        vectors = term_frequency_vectors(psalms, vocabulary, key="lex")

        assert vectors[100].dtype == np.float32


class TestLogCountVectors:
    def test_matches_log1p_of_the_raw_count(self):
        vocabulary = ("A", "B")
        psalms = [_psalm(number=1, lexemes=(("A", "A", "A"),), forms=((),), nodes=(100,))]

        vectors = log_count_vectors(psalms, vocabulary, key="lex")

        assert np.allclose(vectors[100], [np.log1p(3.0), 0.0])

    def test_absent_lexeme_is_log1p_of_zero_which_is_zero(self):
        vocabulary = ("A", "B")
        psalms = [_psalm(number=1, lexemes=(("A",),), forms=((),), nodes=(100,))]

        vectors = log_count_vectors(psalms, vocabulary, key="lex")

        assert vectors[100][1] == 0.0


class TestIcfWeightedVectors:
    def test_scales_binary_presence_by_the_icf_weight(self):
        vocabulary = ("A", "B")
        psalms = [_psalm(number=1, lexemes=(("A", "A"),), forms=((),), nodes=(100,))]
        icf_weights = {"A": 2.5, "B": 0.1}

        vectors = icf_weighted_vectors(psalms, vocabulary, key="lex", icf_weights=icf_weights)

        # binary presence collapses repetition to 1, then scaled by ICF.
        assert np.allclose(vectors[100], [2.5, 0.0])

    def test_absent_lexeme_stays_zero_regardless_of_its_icf_weight(self):
        vocabulary = ("A", "B")
        psalms = [_psalm(number=1, lexemes=(("A",),), forms=((),), nodes=(100,))]
        icf_weights = {"A": 1.0, "B": 99.0}

        vectors = icf_weighted_vectors(psalms, vocabulary, key="lex", icf_weights=icf_weights)

        assert vectors[100][1] == 0.0


class TestTfIcfVectors:
    def test_scales_log_count_by_the_icf_weight(self):
        vocabulary = ("A", "B")
        psalms = [_psalm(number=1, lexemes=(("A", "A", "A"),), forms=((),), nodes=(100,))]
        icf_weights = {"A": 2.0, "B": 5.0}

        vectors = tf_icf_vectors(psalms, vocabulary, key="lex", icf_weights=icf_weights)

        assert np.allclose(vectors[100], [np.log1p(3.0) * 2.0, 0.0])
