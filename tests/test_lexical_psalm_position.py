from __future__ import annotations

import numpy as np

from lexical.corpus import LexicalPsalm
from lexical.positional import bin_index, colon_positions
from lexical.psalm_position import psalm_positional_icf_vectors
from lexical.vectorize import icf_weighted_vectors


def _psalm(*, number, lexemes, forms, nodes):
    return LexicalPsalm(
        number=number,
        colon_lexemes=lexemes,
        colon_forms=forms,
        colon_nodes=nodes,
    )


class TestColonPositionsAndBinIndex:
    def test_reexported_helpers_still_behave_as_before(self):
        t = colon_positions(4)
        assert np.allclose(t, [0.125, 0.375, 0.625, 0.875])
        assert bin_index(t, k=2).tolist() == [0, 0, 1, 1]


class TestPsalmPositionalIcfVectors:
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

        positional = psalm_positional_icf_vectors(
            psalms, vocabulary, key="lex", icf_weights=icf_weights, k=1
        )
        frozen = icf_weighted_vectors(psalms, vocabulary, key="lex", icf_weights=icf_weights)
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

        natural = psalm_positional_icf_vectors(
            psalms, vocabulary, key="lex", icf_weights=icf_weights, k=1
        )
        reversed_order = psalm_positional_icf_vectors(
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
        psalms = [
            _psalm(
                number=1,
                lexemes=(("A",), (), (), ("B",)),
                forms=((), (), (), ()),
                nodes=(100, 101, 102, 103),
            )
        ]

        vectors = psalm_positional_icf_vectors(
            psalms, vocabulary, key="lex", icf_weights=icf_weights, k=2
        )

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
        shuffled = psalm_positional_icf_vectors(
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

        vectors = psalm_positional_icf_vectors(
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

        vectors = psalm_positional_icf_vectors(
            psalms, vocabulary, key="lex", icf_weights=icf_weights, k=4
        )

        assert len(vectors[100]) == 12

    def test_uses_lex0_forms_when_key_is_lex0(self):
        vocabulary = ("A0",)
        icf_weights = {"A0": 5.0}
        psalms = [
            _psalm(number=1, lexemes=(("A",),), forms=(("A0",),), nodes=(100,)),
        ]

        vectors = psalm_positional_icf_vectors(
            psalms, vocabulary, key="lex0", icf_weights=icf_weights, k=1
        )

        assert np.allclose(vectors[100], [5.0])
