from __future__ import annotations

import numpy as np

from lexical.corpus import LexicalPsalm
from lexical.recurrence import lag_bin_index, normalized_lag, spacing_profile_vectors


def _psalm(*, number, lexemes, forms, nodes):
    return LexicalPsalm(
        number=number,
        colon_lexemes=lexemes,
        colon_forms=forms,
        colon_nodes=nodes,
    )


class TestNormalizedLag:
    def test_four_cola_deltas_match_hand_computation(self):
        # triu_indices(4, k=1) order: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        # |i-j|: 1,2,3,1,2,1 -> delta = |i-j|/(4-1)
        delta = normalized_lag(4)

        assert np.allclose(delta, [1 / 3, 2 / 3, 1.0, 1 / 3, 2 / 3, 1 / 3])

    def test_two_cola_gives_a_single_pair_at_max_lag(self):
        delta = normalized_lag(2)

        assert np.allclose(delta, [1.0])


class TestLagBinIndex:
    def test_splits_into_two_bins(self):
        delta = normalized_lag(4)

        bins = lag_bin_index(delta, k=2)

        # [1/3, 2/3, 1.0, 1/3, 2/3, 1/3] -> [0, 1, 1, 0, 1, 0]
        assert bins.tolist() == [0, 1, 1, 0, 1, 0]

    def test_never_exceeds_the_last_bin_at_the_upper_boundary(self):
        bins = lag_bin_index(np.array([1.0]), k=4)

        assert bins.tolist() == [3]


class TestLagProfileVectors:
    def test_vector_length_equals_k(self):
        vocabulary = ("A", "B", "C")
        icf_weights = {"A": 1.0, "B": 1.0, "C": 1.0}
        psalms = [
            _psalm(
                number=1,
                lexemes=(("A",), ("B",), ("C",), ("A",)),
                forms=((), (), (), ()),
                nodes=(100, 101, 102, 103),
            )
        ]

        vectors = spacing_profile_vectors(
            psalms, vocabulary, key="lex", icf_weights=icf_weights, k=4
        )

        assert len(vectors[100]) == 4

    def test_short_range_recurrence_scores_higher_in_the_near_bin(self):
        # A colon repeats its immediate neighbor's vocabulary (adjacent recurrence), and nothing
        # else repeats at long range: near-lag similarity should exceed far-lag similarity.
        vocabulary = ("A", "B", "C", "D", "E", "F")
        icf_weights = dict.fromkeys(vocabulary, 1.0)
        psalms = [
            _psalm(
                number=1,
                lexemes=(("A", "B"), ("A", "B"), ("C",), ("D",), ("E",), ("F",)),
                forms=((), (), (), (), (), ()),
                nodes=(100, 101, 102, 103, 104, 105),
            )
        ]

        vectors = spacing_profile_vectors(
            psalms, vocabulary, key="lex", icf_weights=icf_weights, k=2
        )

        near_bin, far_bin = vectors[100]
        assert near_bin > far_bin

    def test_each_colon_gets_its_own_neighbor_similarity_profile(self):
        # colon 0 (A) matches colon 2 (A) exactly; colon 1 (B) matches neither closely.
        vocabulary = ("A", "B")
        icf_weights = {"A": 1.0, "B": 1.0}
        psalms = [
            _psalm(
                number=1,
                lexemes=(("A",), ("B",), ("A",), ("B",)),
                forms=((), (), (), ()),
                nodes=(100, 101, 102, 103),
            )
        ]

        vectors = spacing_profile_vectors(
            psalms, vocabulary, key="lex", icf_weights=icf_weights, k=2
        )

        assert not np.array_equal(vectors[100], vectors[101])
        assert not np.array_equal(vectors[102], vectors[103])
        # colon 0 and colon 2 both have content "A" but sit at different psalm positions,
        # so their neighbor profiles (computed from their own position outward) differ.
        assert not np.array_equal(vectors[100], vectors[102])

    def test_shuffled_order_changes_the_profile(self):
        vocabulary = ("A", "B", "C", "D", "E", "F")
        icf_weights = dict.fromkeys(vocabulary, 1.0)
        psalms = [
            _psalm(
                number=1,
                lexemes=(("A", "B"), ("A", "B"), ("C",), ("D",), ("E",), ("F",)),
                forms=((), (), (), (), (), ()),
                nodes=(100, 101, 102, 103, 104, 105),
            )
        ]

        natural = spacing_profile_vectors(
            psalms, vocabulary, key="lex", icf_weights=icf_weights, k=2
        )
        shuffled = spacing_profile_vectors(
            psalms,
            vocabulary,
            key="lex",
            icf_weights=icf_weights,
            k=2,
            order_by_psalm={1: np.array([0, 2, 4, 1, 3, 5])},
        )

        assert not np.allclose(natural[100], shuffled[100])

    def test_a_single_colon_psalm_returns_an_all_zero_profile_without_crashing(self):
        vocabulary = ("A",)
        icf_weights = {"A": 1.0}
        psalms = [_psalm(number=1, lexemes=(("A",),), forms=((),), nodes=(100,))]

        vectors = spacing_profile_vectors(
            psalms, vocabulary, key="lex", icf_weights=icf_weights, k=4
        )

        assert np.allclose(vectors[100], [0.0, 0.0, 0.0, 0.0])

    def test_a_colon_with_no_vocabulary_matches_does_not_produce_nan(self):
        vocabulary = ("A", "B")
        icf_weights = {"A": 1.0, "B": 1.0}
        psalms = [
            _psalm(
                number=1,
                lexemes=(("A",), ("OUT_OF_VOCAB",)),
                forms=((), ()),
                nodes=(100, 101),
            )
        ]

        vectors = spacing_profile_vectors(
            psalms, vocabulary, key="lex", icf_weights=icf_weights, k=2
        )

        assert not np.isnan(vectors[100]).any()

    def test_uses_lex0_forms_when_key_is_lex0(self):
        vocabulary = ("A0", "B0")
        icf_weights = {"A0": 1.0, "B0": 1.0}
        psalms = [
            _psalm(
                number=1,
                lexemes=(("A",), ("B",)),
                forms=(("A0",), ("B0",)),
                nodes=(100, 101),
            )
        ]

        vectors = spacing_profile_vectors(
            psalms, vocabulary, key="lex0", icf_weights=icf_weights, k=2
        )

        assert len(vectors[100]) == 2
