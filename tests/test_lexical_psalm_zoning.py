from __future__ import annotations

import numpy as np

from lexical.corpus import LexicalPsalm
from lexical.psalm_zoning import psalm_positional_centroid_vectors


def _psalm(*, number, lexemes, forms, nodes):
    return LexicalPsalm(
        number=number,
        half_verse_lexemes=lexemes,
        half_verse_forms=forms,
        half_verse_nodes=nodes,
    )


class TestPsalmPositionalCentroidVectors:
    def test_vector_length_is_twice_the_vocabulary_size(self):
        vocabulary = ("A", "B", "C")
        icf_weights = {"A": 1.0, "B": 1.0, "C": 1.0}
        psalms = [_psalm(number=1, lexemes=(("A",), ("B",)), forms=((), ()), nodes=(100, 101))]

        vectors = psalm_positional_centroid_vectors(
            psalms, vocabulary, key="lex", icf_weights=icf_weights
        )

        assert len(vectors[100]) == 6

    def test_inventory_half_is_icf_weight_if_present_anywhere_in_the_psalm(self):
        vocabulary = ("A", "B", "C")
        icf_weights = {"A": 2.0, "B": 3.0, "C": 5.0}
        psalms = [
            _psalm(
                number=1,
                lexemes=(("A",), ("A", "B"), ("A",)),
                forms=((), (), ()),
                nodes=(100, 101, 102),
            )
        ]

        vectors = psalm_positional_centroid_vectors(
            psalms, vocabulary, key="lex", icf_weights=icf_weights
        )

        b = vectors[100][:3]
        assert np.allclose(b, [2.0, 3.0, 0.0])

    def test_single_occurrence_lexeme_has_centroid_equal_to_its_own_colon_position(self):
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

        vectors = psalm_positional_centroid_vectors(
            psalms, vocabulary, key="lex", icf_weights=icf_weights
        )

        m = vectors[100][1]
        expected_mu = 0.125
        assert np.isclose(m, 4.0 * (2 * expected_mu - 1))

    def test_lexeme_present_in_every_colon_is_centered_near_zero(self):
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

        vectors = psalm_positional_centroid_vectors(
            psalms, vocabulary, key="lex", icf_weights=icf_weights
        )

        m = vectors[100][1]
        assert np.isclose(m, 0.0)

    def test_early_leaning_lexeme_has_negative_m_and_late_leaning_has_positive_m(self):
        vocabulary = ("EARLY", "LATE")
        icf_weights = {"EARLY": 1.0, "LATE": 1.0}
        psalms = [
            _psalm(
                number=1,
                lexemes=(("EARLY",), (), (), ("LATE",)),
                forms=((), (), (), ()),
                nodes=(100, 101, 102, 103),
            )
        ]

        vectors = psalm_positional_centroid_vectors(
            psalms, vocabulary, key="lex", icf_weights=icf_weights
        )

        m_early, m_late = vectors[100][2], vectors[100][3]
        assert m_early < 0
        assert m_late > 0

    def test_absent_lexeme_is_zero_in_both_halves(self):
        vocabulary = ("A", "B")
        icf_weights = {"A": 1.0, "B": 99.0}
        psalms = [_psalm(number=1, lexemes=(("A",),), forms=((),), nodes=(100,))]

        vectors = psalm_positional_centroid_vectors(
            psalms, vocabulary, key="lex", icf_weights=icf_weights
        )

        assert vectors[100][1] == 0.0
        assert vectors[100][3] == 0.0

    def test_broadcasts_the_same_psalm_level_vector_to_every_colon_node(self):
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

        vectors = psalm_positional_centroid_vectors(
            psalms, vocabulary, key="lex", icf_weights=icf_weights
        )

        assert np.array_equal(vectors[100], vectors[101])
        assert np.array_equal(vectors[101], vectors[102])

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

        natural = psalm_positional_centroid_vectors(
            psalms, vocabulary, key="lex", icf_weights=icf_weights
        )
        shuffled = psalm_positional_centroid_vectors(
            psalms,
            vocabulary,
            key="lex",
            icf_weights=icf_weights,
            order_by_psalm={1: np.array([3, 1, 2, 0])},
        )

        assert natural[100][0] == shuffled[100][0]
        assert natural[100][1] != shuffled[100][1]

    def test_uses_lex0_forms_when_key_is_lex0(self):
        vocabulary = ("A0",)
        icf_weights = {"A0": 1.0}
        psalms = [_psalm(number=1, lexemes=(("A",),), forms=(("A0",),), nodes=(100,))]

        vectors = psalm_positional_centroid_vectors(
            psalms, vocabulary, key="lex0", icf_weights=icf_weights
        )

        assert len(vectors[100]) == 2
