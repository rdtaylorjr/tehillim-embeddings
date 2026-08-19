from __future__ import annotations

import numpy as np

from lexical.corpus import LexicalPsalm
from lexical.positional import bin_index, colon_positions, positional_icf_vectors
from lexical.vectorize import icf_weighted_vectors


def _psalm(*, number, lexemes, forms, nodes):
    return LexicalPsalm(
        number=number,
        half_verse_lexemes=lexemes,
        half_verse_forms=forms,
        half_verse_nodes=nodes,
    )


class TestColonPositions:
    def test_continuity_corrected_midpoints_for_four_cola(self):
        t = colon_positions(4)

        assert np.allclose(t, [0.125, 0.375, 0.625, 0.875])

    def test_single_colon_sits_at_the_center(self):
        t = colon_positions(1)

        assert np.allclose(t, [0.5])


class TestBinIndex:
    def test_splits_four_cola_into_two_halves(self):
        t = colon_positions(4)

        bins = bin_index(t, k=2)

        assert bins.tolist() == [0, 0, 1, 1]

    def test_splits_four_cola_into_four_quarters(self):
        t = colon_positions(4)

        bins = bin_index(t, k=4)

        assert bins.tolist() == [0, 1, 2, 3]

    def test_never_exceeds_the_last_bin_at_the_upper_boundary(self):
        t = np.array([1.0])  # a boundary value, not actually produced by colon_positions

        bins = bin_index(t, k=4)

        assert bins.tolist() == [3]


class TestPositionalIcfVectors:
    def test_k1_matches_the_frozen_unordered_icf_representation_exactly(self):
        vocabulary = ("A", "B", "C")
        icf_weights = {"A": 2.0, "B": 0.5, "C": 1.0}
        psalms = [
            _psalm(
                number=1,
                lexemes=(("A",), ("B", "C"), ("A", "A")),
                forms=((), (), ()),
                nodes=(100, 101, 102),
            )
        ]

        positional = positional_icf_vectors(
            psalms, vocabulary, key="lex", icf_weights=icf_weights, k=1
        )
        frozen = icf_weighted_vectors(psalms, vocabulary, key="lex", icf_weights=icf_weights)
        # frozen is per-colon; the psalm-level k=1 sum must equal the sum of all its colons
        # (cosine similarity is scale-invariant, so sum vs. mean-pooled centroid rank identically).
        expected = sum(frozen[node] for node in (100, 101, 102))

        for node in (100, 101, 102):
            assert np.allclose(positional[node], expected)

    def test_k1_is_invariant_under_any_colon_order_since_it_is_a_sum(self):
        vocabulary = ("A", "B", "C")
        icf_weights = {"A": 2.0, "B": 0.5, "C": 1.0}
        psalms = [
            _psalm(
                number=1,
                lexemes=(("A",), ("B", "C"), ("A", "A")),
                forms=((), (), ()),
                nodes=(100, 101, 102),
            )
        ]

        natural = positional_icf_vectors(
            psalms, vocabulary, key="lex", icf_weights=icf_weights, k=1
        )
        reversed_order = positional_icf_vectors(
            psalms,
            vocabulary,
            key="lex",
            icf_weights=icf_weights,
            k=1,
            order_by_psalm={1: np.array([2, 1, 0])},
        )

        assert np.allclose(natural[100], reversed_order[100])

    def test_k2_splits_content_between_the_first_and_second_half(self):
        vocabulary = ("A", "B")
        icf_weights = {"A": 2.0, "B": 3.0}
        # 4 cola: A only in colon 0, B only in colon 3 (per colon_positions/bin_index above,
        # cola 0-1 fall in bin 0, cola 2-3 fall in bin 1).
        psalms = [
            _psalm(
                number=1,
                lexemes=(("A",), (), (), ("B",)),
                forms=((), (), (), ()),
                nodes=(100, 101, 102, 103),
            )
        ]

        vectors = positional_icf_vectors(
            psalms, vocabulary, key="lex", icf_weights=icf_weights, k=2
        )

        # [B_1; B_2] = [A_weight, 0, 0, B_weight]
        assert np.allclose(vectors[100], [2.0, 0.0, 0.0, 3.0])

    def test_shuffled_order_moves_content_between_bins(self):
        vocabulary = ("A", "B")
        icf_weights = {"A": 2.0, "B": 3.0}
        psalms = [
            _psalm(
                number=1,
                lexemes=(("A",), (), (), ("B",)),
                forms=((), (), (), ()),
                nodes=(100, 101, 102, 103),
            )
        ]
        # Reverse the colon content order: now A's content lands in position 3 (bin 1) and
        # B's content lands in position 0 (bin 0), so B is in the first block, A in the second.
        shuffled = positional_icf_vectors(
            psalms,
            vocabulary,
            key="lex",
            icf_weights=icf_weights,
            k=2,
            order_by_psalm={1: np.array([3, 2, 1, 0])},
        )

        assert np.allclose(shuffled[100], [0.0, 3.0, 2.0, 0.0])

    def test_broadcasts_the_same_psalm_level_vector_to_every_colon_node(self):
        vocabulary = ("A", "B")
        icf_weights = {"A": 2.0, "B": 3.0}
        psalms = [
            _psalm(
                number=1,
                lexemes=(("A",), (), (), ("B",)),
                forms=((), (), (), ()),
                nodes=(100, 101, 102, 103),
            )
        ]

        vectors = positional_icf_vectors(
            psalms, vocabulary, key="lex", icf_weights=icf_weights, k=2
        )

        assert np.array_equal(vectors[100], vectors[101])
        assert np.array_equal(vectors[101], vectors[102])
        assert np.array_equal(vectors[102], vectors[103])

    def test_vector_length_is_k_times_vocabulary_size(self):
        vocabulary = ("A", "B", "C")
        icf_weights = {"A": 1.0, "B": 1.0, "C": 1.0}
        psalms = [
            _psalm(number=1, lexemes=(("A",), ("B",)), forms=((), ()), nodes=(100, 101)),
        ]

        vectors = positional_icf_vectors(
            psalms, vocabulary, key="lex", icf_weights=icf_weights, k=4
        )

        assert len(vectors[100]) == 12

    def test_uses_lex0_forms_when_key_is_lex0(self):
        vocabulary = ("A0",)
        icf_weights = {"A0": 5.0}
        psalms = [
            _psalm(number=1, lexemes=(("A",),), forms=(("A0",),), nodes=(100,)),
        ]

        vectors = positional_icf_vectors(
            psalms, vocabulary, key="lex0", icf_weights=icf_weights, k=1
        )

        assert np.allclose(vectors[100], [5.0])
