from __future__ import annotations

import numpy as np
import pytest

from syntactic.assignment import Assignment
from syntactic.clause_tab import (
    TAB_TRANSITION_VOCABULARY,
    TAB_VOCABULARY,
    clause_tab_1gram_vectors,
    clause_tab_columns,
    clause_tab_transition_columns,
    clause_tab_transition_psalm_vectors,
    tab_label,
    tab_transition_label,
)
from syntactic.corpus import ClausePsalm


def _psalm(nodes, tab, node_index, unit_index, weight):
    n = len(tab)
    return ClausePsalm(
        number=1,
        half_verse_nodes=nodes,
        clause_atom_nodes=tuple(range(n)),
        clause_atom_tab=tab,
        clause_atom_assignment=Assignment(
            node_index=np.array(node_index),
            unit_index=np.array(unit_index),
            weight=np.array(weight),
        ),
    )


class TestTabLabel:
    @pytest.mark.parametrize(("depth", "expected"), [(0, "0"), (5, "5"), (10, "10"), (11, "11+")])
    def test_caps_the_thin_tail(self, depth, expected):
        assert tab_label(depth) == expected

    def test_the_whole_bible_maximum_still_lands_in_the_cap(self):
        """BHSA reaches 29, so a Psalms-fitted vocabulary would break on another corpus."""
        assert tab_label(29) == "11+"

    def test_every_label_is_in_the_vocabulary(self):
        assert {tab_label(d) for d in range(30)} <= set(TAB_VOCABULARY)


class TestTabTransitionLabel:
    @pytest.mark.parametrize(
        ("change", "expected"),
        [(-11, "<=-3"), (-3, "<=-3"), (-2, "-2"), (-1, "-1"), (0, "0"), (1, "+1"), (4, "+4")],
    )
    def test_labels_signed_steps(self, change, expected):
        assert tab_transition_label(change) == expected

    def test_caps_both_directions(self):
        assert tab_transition_label(5) == ">=+5"
        assert tab_transition_label(99) == ">=+5"

    def test_every_label_is_in_the_vocabulary(self):
        assert {tab_transition_label(c) for c in range(-15, 15)} <= set(TAB_TRANSITION_VOCABULARY)


class TestColumns:
    def test_depths_group_into_their_colon(self):
        psalm = _psalm((10, 11), (3, 4, 7), [0, 1, 2], [0, 0, 1], [1.0, 1.0, 1.0])

        assert clause_tab_columns(psalm) == (("3", "4"), ("7",))

    def test_a_transition_is_the_change_between_consecutive_atoms(self):
        psalm = _psalm((10,), (3, 4, 2), [0, 1, 2], [0, 0, 0], [1.0, 1.0, 1.0])

        assert clause_tab_transition_columns(psalm) == (("+1", "-2"),)

    def test_a_single_atom_colon_yields_no_transition(self):
        psalm = _psalm((10,), (5,), [0], [0], [1.0])

        assert clause_tab_transition_columns(psalm) == ((),)

    def test_transitions_are_computed_before_capping_so_a_step_is_the_real_change(self):
        """Capping depths first would turn a 12-to-14 step into no change at all."""
        psalm = _psalm((10,), (12, 14), [0, 1], [0, 0], [1.0, 1.0])

        assert clause_tab_columns(psalm) == (("11+", "11+"),)
        assert clause_tab_transition_columns(psalm) == (("+2",),)


class TestTransitionPermutation:
    """The order lands on the depths, since permuting differences reorders no real sequence."""

    def test_the_permutation_moves_the_contour(self):
        psalm = _psalm((10,), (3, 4, 2), [0, 1, 2], [0, 0, 0], [1.0, 1.0, 1.0])

        permuted = clause_tab_transition_columns(psalm, {10: np.array([2, 0, 1])})

        assert clause_tab_transition_columns(psalm) == (("+1", "-2"),)
        assert permuted == (("+1", "+1"),)

    def test_the_permutation_keeps_every_step_the_real_contour_had(self):
        """A permutation sized at transition length would truncate the depths and drop steps."""
        psalm = _psalm((10,), (3, 4, 2), [0, 1, 2], [0, 0, 0], [1.0, 1.0, 1.0])

        permuted = clause_tab_transition_columns(psalm, {10: np.array([2, 0, 1])})

        assert len(permuted[0]) == len(clause_tab_transition_columns(psalm)[0])

    def test_a_permutation_sized_at_the_transitions_is_refused(self):
        psalm = _psalm((10,), (3, 4, 2), [0, 1, 2], [0, 0, 0], [1.0, 1.0, 1.0])

        with pytest.raises(ValueError, match="permutation of length 2"):
            clause_tab_transition_columns(psalm, {10: np.array([1, 0])})


class TestVectors:
    def test_an_inventory_histogram_is_the_depth_proportions(self):
        psalm = _psalm((10,), (4, 4, 5), [0, 1, 2], [0, 0, 0], [1.0, 1.0, 1.0])

        vector = clause_tab_1gram_vectors([psalm])[10]

        assert vector[TAB_VOCABULARY.index("4")] == pytest.approx(2 / 3)
        assert vector[TAB_VOCABULARY.index("5")] == pytest.approx(1 / 3)

    def test_the_inventory_width_is_the_capped_vocabulary(self):
        psalm = _psalm((10,), (0,), [0], [0], [1.0])

        assert clause_tab_1gram_vectors([psalm])[10].shape == (len(TAB_VOCABULARY),)

    def test_the_transition_vector_is_psalm_broadcast(self):
        psalm = _psalm((10, 11), (1, 2, 5), [0, 1, 2], [0, 0, 1], [1.0, 1.0, 1.0])

        vectors = clause_tab_transition_psalm_vectors([psalm])

        assert vectors[10].tolist() == vectors[11].tolist()
        assert vectors[10][TAB_TRANSITION_VOCABULARY.index("+1")] == pytest.approx(1.0)
