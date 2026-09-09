from __future__ import annotations

import numpy as np
import pytest

from syntactic.assignment import (
    Assignment,
    assign_nodes_to_units,
    containment_mask,
    majority_mask,
)


def _csr(node_slot_lists):
    slots = np.concatenate([np.asarray(s, dtype=np.int32) for s in node_slot_lists])
    counts = np.asarray([len(s) for s in node_slot_lists], dtype=np.int32)
    return slots, counts


class TestAssignNodesToUnits:
    def test_a_node_wholly_inside_one_unit_gets_weight_one(self):
        slot_unit = np.array([-1, 0, 0, 1, 1], dtype=np.int32)
        slots, counts = _csr([[1, 2]])

        result = assign_nodes_to_units(slot_unit, slots, counts, n_units=2)

        assert result.node_index.tolist() == [0]
        assert result.unit_index.tolist() == [0]
        assert result.weight.tolist() == [1.0]

    def test_a_node_split_across_two_units_splits_its_weight_by_word_count(self):
        slot_unit = np.array([-1, 0, 0, 0, 1], dtype=np.int32)
        slots, counts = _csr([[1, 2, 3, 4]])

        result = assign_nodes_to_units(slot_unit, slots, counts, n_units=2)

        assert result.node_index.tolist() == [0, 0]
        assert result.unit_index.tolist() == [0, 1]
        assert result.weight.tolist() == pytest.approx([0.75, 0.25])

    def test_weights_of_one_node_sum_to_one_so_assignment_conserves_mass(self):
        slot_unit = np.array([-1, 0, 1, 2, 2], dtype=np.int32)
        slots, counts = _csr([[1, 2, 3, 4]])

        result = assign_nodes_to_units(slot_unit, slots, counts, n_units=3)

        assert result.weight.sum() == pytest.approx(1.0)

    def test_slots_outside_every_unit_are_dropped_without_shrinking_the_denominator(self):
        #: A word outside the psalm's half verses must not inflate the weight of the rest.
        slot_unit = np.array([-1, 0, 0, -1, -1], dtype=np.int32)
        slots, counts = _csr([[1, 2, 3, 4]])

        result = assign_nodes_to_units(slot_unit, slots, counts, n_units=1)

        assert result.weight.tolist() == pytest.approx([0.5])

    def test_pairs_are_ordered_by_node_then_unit(self):
        slot_unit = np.array([-1, 1, 0, 0, 1], dtype=np.int32)
        slots, counts = _csr([[1, 2], [3, 4]])

        result = assign_nodes_to_units(slot_unit, slots, counts, n_units=2)

        assert list(zip(result.node_index.tolist(), result.unit_index.tolist(), strict=True)) == [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]

    def test_a_node_with_no_slot_in_any_unit_produces_no_pair(self):
        slot_unit = np.array([-1, -1, 0], dtype=np.int32)
        slots, counts = _csr([[0, 1], [2]])

        result = assign_nodes_to_units(slot_unit, slots, counts, n_units=1)

        assert result.node_index.tolist() == [1]

    def test_no_nodes_yields_empty_arrays_rather_than_raising(self):
        slot_unit = np.array([-1, 0], dtype=np.int32)

        result = assign_nodes_to_units(
            slot_unit, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), n_units=1
        )

        assert result.node_index.size == 0
        assert result.weight.size == 0

    def test_a_zero_slot_node_is_rejected_rather_than_dividing_by_zero(self):
        slot_unit = np.array([-1, 0], dtype=np.int32)
        slots = np.array([1], dtype=np.int32)
        counts = np.array([0, 1], dtype=np.int32)

        with pytest.raises(ValueError, match="no slots"):
            assign_nodes_to_units(slot_unit, slots, counts, n_units=1)


class TestContainmentMask:
    def test_selects_only_pairs_whose_node_lies_wholly_in_the_unit(self):
        assignment = Assignment(
            node_index=np.array([0, 0, 1]),
            unit_index=np.array([0, 1, 1]),
            weight=np.array([0.75, 0.25, 1.0]),
        )

        assert containment_mask(assignment).tolist() == [False, False, True]

    def test_tolerates_float_error_in_a_weight_that_should_be_exactly_one(self):
        assignment = Assignment(
            node_index=np.array([0]), unit_index=np.array([0]), weight=np.array([1.0 - 1e-12])
        )

        assert containment_mask(assignment).tolist() == [True]


class TestMajorityMask:
    def test_selects_the_unit_holding_most_of_a_split_nodes_words(self):
        assignment = Assignment(
            node_index=np.array([0, 0]),
            unit_index=np.array([0, 1]),
            weight=np.array([0.25, 0.75]),
        )

        assert majority_mask(assignment).tolist() == [False, True]

    def test_a_wholly_contained_node_selects_its_only_unit(self):
        assignment = Assignment(
            node_index=np.array([0]), unit_index=np.array([1]), weight=np.array([1.0])
        )

        assert majority_mask(assignment).tolist() == [True]

    def test_an_even_split_goes_to_the_earlier_unit_so_the_rule_is_deterministic(self):
        assignment = Assignment(
            node_index=np.array([0, 0]),
            unit_index=np.array([2, 1]),
            weight=np.array([0.5, 0.5]),
        )

        selected = assignment.unit_index[majority_mask(assignment)]

        assert selected.tolist() == [1]

    def test_every_node_is_selected_exactly_once(self):
        assignment = Assignment(
            node_index=np.array([0, 0, 1, 2, 2, 2]),
            unit_index=np.array([0, 1, 1, 0, 1, 2]),
            weight=np.array([0.4, 0.6, 1.0, 0.2, 0.5, 0.3]),
        )

        mask = majority_mask(assignment)

        assert np.bincount(assignment.node_index[mask], minlength=3).tolist() == [1, 1, 1]

    def test_no_nodes_yields_an_empty_mask(self):
        empty = np.empty(0, dtype=np.int64)
        assignment = Assignment(node_index=empty, unit_index=empty, weight=np.empty(0))

        assert majority_mask(assignment).tolist() == []
