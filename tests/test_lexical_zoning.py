from __future__ import annotations

import numpy as np

from lexical.corpus import LexicalPsalm
from lexical.zoning import position_mean_vectors


def _psalm(*, number, lexemes, forms, nodes):
    return LexicalPsalm(
        number=number,
        colon_lexemes=lexemes,
        colon_forms=forms,
        colon_nodes=nodes,
    )


class TestPositionalCentroidVectors:
    def test_vector_length_is_twice_the_vocabulary_size(self):
        vocabulary = ("A", "B", "C")
        icf_weights = {"A": 1.0, "B": 1.0, "C": 1.0}
        psalms = [_psalm(number=1, lexemes=(("A",), ("B",)), forms=((), ()), nodes=(100, 101))]

        vectors = position_mean_vectors(psalms, vocabulary, key="lex", icf_weights=icf_weights)

        assert len(vectors[100]) == 6

    def test_inventory_half_is_icf_weight_if_present_in_that_colon(self):
        vocabulary = ("A", "B", "C")
        icf_weights = {"A": 2.0, "B": 3.0, "C": 5.0}
        # A appears in every colon, B only in colon 1, C absent.
        psalms = [
            _psalm(
                number=1,
                lexemes=(("A",), ("A", "B"), ("A",)),
                forms=((), (), ()),
                nodes=(100, 101, 102),
            )
        ]

        vectors = position_mean_vectors(psalms, vocabulary, key="lex", icf_weights=icf_weights)

        # colon 0 has only A: b reflects this colon's own content, not the whole psalm's.
        assert np.allclose(vectors[100][:3], [2.0, 0.0, 0.0])
        assert np.allclose(vectors[101][:3], [2.0, 3.0, 0.0])

    def test_m_equals_the_colons_own_position_scaled_by_icf_weight(self):
        vocabulary = ("A",)
        icf_weights = {"A": 4.0}
        # 4 cola, A only in colon index 0 -> t_0 = (1-0.5)/4 = 0.125
        psalms = [
            _psalm(
                number=1,
                lexemes=(("A",), (), (), ()),
                forms=((), (), (), ()),
                nodes=(100, 101, 102, 103),
            )
        ]

        vectors = position_mean_vectors(psalms, vocabulary, key="lex", icf_weights=icf_weights)

        m = vectors[100][1]
        expected_t = 0.125
        assert np.isclose(m, 4.0 * (2 * expected_t - 1))

    def test_the_same_lexeme_gets_a_different_m_per_colon_matching_each_ones_own_position(self):
        vocabulary = ("A",)
        icf_weights = {"A": 4.0}
        psalms = [
            _psalm(
                number=1,
                lexemes=(("A",), ("A",), ("A",), ("A",)),
                forms=((), (), (), ()),
                nodes=(100, 101, 102, 103),
            )
        ]

        vectors = position_mean_vectors(psalms, vocabulary, key="lex", icf_weights=icf_weights)

        # symmetric colon positions -> symmetric signed m values, distinct per colon.
        assert vectors[100][1] < 0.0
        assert vectors[103][1] > 0.0
        assert not np.isclose(vectors[100][1], vectors[101][1])

    def test_early_leaning_lexeme_has_negative_m_and_late_leaning_has_positive_m(self):
        vocabulary = ("EARLY", "LATE")
        icf_weights = {"EARLY": 1.0, "LATE": 1.0}
        # 4 cola: EARLY only in colon 0, LATE only in colon 3.
        psalms = [
            _psalm(
                number=1,
                lexemes=(("EARLY",), (), (), ("LATE",)),
                forms=((), (), (), ()),
                nodes=(100, 101, 102, 103),
            )
        ]

        vectors = position_mean_vectors(psalms, vocabulary, key="lex", icf_weights=icf_weights)

        m_early = vectors[100][2]
        m_late = vectors[103][3]
        assert m_early < 0
        assert m_late > 0

    def test_absent_lexeme_is_zero_in_both_halves(self):
        vocabulary = ("A", "B")
        icf_weights = {"A": 1.0, "B": 99.0}
        psalms = [_psalm(number=1, lexemes=(("A",),), forms=((),), nodes=(100,))]

        vectors = position_mean_vectors(psalms, vocabulary, key="lex", icf_weights=icf_weights)

        assert vectors[100][1] == 0.0  # b for B
        assert vectors[100][3] == 0.0  # m for B

    def test_each_colon_gets_its_own_distinguishable_vector(self):
        vocabulary = ("A", "B")
        icf_weights = {"A": 1.0, "B": 1.0}
        psalms = [
            _psalm(
                number=1,
                lexemes=(("A",), ("B",), ("A",)),
                forms=((), (), ()),
                nodes=(100, 101, 102),
            )
        ]

        vectors = position_mean_vectors(psalms, vocabulary, key="lex", icf_weights=icf_weights)

        assert not np.array_equal(vectors[100], vectors[101])
        # colon 0 and colon 2 share content (A) but sit at different positions.
        assert not np.array_equal(vectors[100], vectors[102])

    def test_inventory_half_is_invariant_under_shuffled_order_but_position_half_is_not(self):
        vocabulary = ("A",)
        icf_weights = {"A": 4.0}
        psalms = [
            _psalm(
                number=1,
                lexemes=(("A",), (), (), ()),
                forms=((), (), (), ()),
                nodes=(100, 101, 102, 103),
            )
        ]

        natural = position_mean_vectors(psalms, vocabulary, key="lex", icf_weights=icf_weights)
        shuffled = position_mean_vectors(
            psalms,
            vocabulary,
            key="lex",
            icf_weights=icf_weights,
            order_by_psalm={1: np.array([3, 1, 2, 0])},  # moves colon 0 to position 3
        )

        assert natural[100][0] == shuffled[100][0]  # b: invariant, same colon's own content
        assert natural[100][1] != shuffled[100][1]  # m: not invariant, colon's position moved

    def test_uses_lex0_forms_when_key_is_lex0(self):
        vocabulary = ("A0",)
        icf_weights = {"A0": 1.0}
        psalms = [_psalm(number=1, lexemes=(("A",),), forms=(("A0",),), nodes=(100,))]

        vectors = position_mean_vectors(psalms, vocabulary, key="lex0", icf_weights=icf_weights)

        assert len(vectors[100]) == 2
