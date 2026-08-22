from __future__ import annotations

import numpy as np

from morphological.corpus import MorphologicalPsalm
from morphological.deploy import psalm_deploy_vectors, suffix_deploy_vectors
from morphological.suffix import NONE_SUFFIX_TOKEN, SUFFIX_VOCABULARY


def _psalm(*, number, nodes, **feature_columns):
    return MorphologicalPsalm(
        number=number,
        half_verse_nodes=nodes,
        **{f"half_verse_{feature}": values for feature, values in feature_columns.items()},
    )


def _colon_values(*colons):
    return lambda psalm: colons


class TestPsalmDeployVectors:
    def test_vector_length_is_twice_the_vocabulary_size(self):
        vocabulary = ("A", "B", "C")
        psalms = [_psalm(number=1, nodes=(100, 101))]
        colon_values = _colon_values(("A",), ("B",))

        vectors = psalm_deploy_vectors(psalms, vocabulary, colon_values)

        assert len(vectors[100]) == 6

    def test_inventory_half_is_one_if_present_anywhere_in_the_psalm_else_zero(self):
        vocabulary = ("A", "B", "C")
        psalms = [_psalm(number=1, nodes=(100, 101, 102))]
        colon_values = _colon_values(("A",), ("A", "B"), ("A",))

        vectors = psalm_deploy_vectors(psalms, vocabulary, colon_values)

        b = vectors[100][:3]
        assert np.allclose(b, [1.0, 1.0, 0.0])

    def test_single_occurrence_value_has_centroid_equal_to_its_own_colon_position(self):
        vocabulary = ("A",)
        psalms = [_psalm(number=1, nodes=(100, 101, 102, 103))]
        colon_values = _colon_values(("A",), (), (), ())

        vectors = psalm_deploy_vectors(psalms, vocabulary, colon_values)

        m = vectors[100][1]
        expected_mu = 0.125
        assert np.isclose(m, 2 * expected_mu - 1)

    def test_value_present_in_every_colon_is_centered_near_zero(self):
        vocabulary = ("A",)
        psalms = [_psalm(number=1, nodes=(100, 101, 102, 103))]
        colon_values = _colon_values(("A",), ("A",), ("A",), ("A",))

        vectors = psalm_deploy_vectors(psalms, vocabulary, colon_values)

        assert np.isclose(vectors[100][1], 0.0)

    def test_early_leaning_value_has_negative_m_and_late_leaning_has_positive_m(self):
        vocabulary = ("EARLY", "LATE")
        psalms = [_psalm(number=1, nodes=(100, 101, 102, 103))]
        colon_values = _colon_values(("EARLY",), (), (), ("LATE",))

        vectors = psalm_deploy_vectors(psalms, vocabulary, colon_values)

        m_early, m_late = vectors[100][2], vectors[100][3]
        assert m_early < 0
        assert m_late > 0

    def test_absent_value_is_zero_in_both_halves(self):
        vocabulary = ("A", "B")
        psalms = [_psalm(number=1, nodes=(100,))]
        colon_values = _colon_values(("A",))

        vectors = psalm_deploy_vectors(psalms, vocabulary, colon_values)

        assert vectors[100][1] == 0.0
        assert vectors[100][3] == 0.0

    def test_broadcasts_the_same_psalm_level_vector_to_every_colon_node(self):
        vocabulary = ("A", "B")
        psalms = [_psalm(number=1, nodes=(100, 101, 102))]
        colon_values = _colon_values(("A",), ("B",), ("A",))

        vectors = psalm_deploy_vectors(psalms, vocabulary, colon_values)

        assert np.array_equal(vectors[100], vectors[101])
        assert np.array_equal(vectors[101], vectors[102])

    def test_inventory_half_is_invariant_under_shuffled_order_but_position_half_is_not(self):
        vocabulary = ("A",)
        psalms = [_psalm(number=1, nodes=(100, 101, 102, 103))]
        colon_values = _colon_values(("A",), (), (), ())

        natural = psalm_deploy_vectors(psalms, vocabulary, colon_values)
        shuffled = psalm_deploy_vectors(
            psalms, vocabulary, colon_values, order_by_psalm={1: np.array([3, 1, 2, 0])}
        )

        assert natural[100][0] == shuffled[100][0]
        assert natural[100][1] != shuffled[100][1]


class TestSuffixDeployVectors:
    def test_dimension_is_twice_the_suffix_vocabulary(self):
        psalm = _psalm(
            number=1,
            nodes=(100, 101),
            prs_gn=(("NA",), ("m",)),
            prs_nu=(("NA",), ("pl",)),
            prs_ps=(("NA",), ("p3",)),
        )

        vectors = suffix_deploy_vectors([psalm])

        assert len(vectors[100]) == 2 * len(SUFFIX_VOCABULARY)

    def test_none_suffix_present_in_every_colon_of_a_suffixless_psalm(self):
        psalm = _psalm(
            number=1,
            nodes=(100, 101),
            prs_gn=(("NA",), ("NA",)),
            prs_nu=(("NA",), ("NA",)),
            prs_ps=(("NA",), ("NA",)),
        )

        vectors = suffix_deploy_vectors([psalm])

        none_index = SUFFIX_VOCABULARY.index(NONE_SUFFIX_TOKEN)
        dim = len(SUFFIX_VOCABULARY)
        assert vectors[100][none_index] == 1.0
        assert np.isclose(vectors[100][dim + none_index], 0.0)
