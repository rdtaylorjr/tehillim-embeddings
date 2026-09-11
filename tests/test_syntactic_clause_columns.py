from __future__ import annotations

import numpy as np

from syntactic.assignment import Assignment, containment_mask, majority_mask
from syntactic.clause_columns import colon_sequences


def _assignment(node_index, unit_index, weight):
    return Assignment(
        node_index=np.array(node_index),
        unit_index=np.array(unit_index),
        weight=np.array(weight),
    )


class TestColonSequences:
    def test_groups_each_colons_values_in_corpus_node_order(self):
        assignment = _assignment([0, 1, 2], [0, 0, 1], [1.0, 1.0, 1.0])

        result = colon_sequences(
            ("a", "b", "c"), assignment, n_colons=2, mask=majority_mask(assignment)
        )

        assert result == (("a", "b"), ("c",))

    def test_a_colon_no_node_reaches_gets_an_empty_sequence(self):
        assignment = _assignment([0], [0], [1.0])

        result = colon_sequences(("a",), assignment, n_colons=3, mask=majority_mask(assignment))

        assert result == (("a",), (), ())

    def test_majority_places_a_split_node_in_exactly_one_colon(self):
        assignment = _assignment([0, 0, 1], [0, 1, 1], [0.25, 0.75, 1.0])

        result = colon_sequences(
            ("split", "whole"), assignment, n_colons=2, mask=majority_mask(assignment)
        )

        assert result == ((), ("split", "whole"))

    def test_containment_drops_a_split_node_from_both_colons(self):
        assignment = _assignment([0, 0, 1], [0, 1, 1], [0.25, 0.75, 1.0])

        result = colon_sequences(
            ("split", "whole"), assignment, n_colons=2, mask=containment_mask(assignment)
        )

        assert result == ((), ("whole",))

    def test_node_order_is_corpus_order_not_the_order_pairs_happen_to_appear(self):
        assignment = _assignment([2, 0, 1], [0, 0, 0], [1.0, 1.0, 1.0])

        result = colon_sequences(
            ("first", "second", "third"), assignment, n_colons=1, mask=majority_mask(assignment)
        )

        assert result == (("first", "second", "third"),)

    def test_an_empty_assignment_gives_every_colon_an_empty_sequence(self):
        empty = np.empty(0, dtype=np.int64)
        assignment = Assignment(node_index=empty, unit_index=empty, weight=np.empty(0))

        result = colon_sequences((), assignment, n_colons=2, mask=np.empty(0, dtype=bool))

        assert result == ((), ())

    def test_every_majority_selected_node_appears_exactly_once_across_the_colons(self):
        assignment = _assignment([0, 0, 1, 2], [0, 1, 1, 0], [0.4, 0.6, 1.0, 1.0])

        result = colon_sequences(
            ("a", "b", "c"), assignment, n_colons=2, mask=majority_mask(assignment)
        )

        assert sorted(v for colon in result for v in colon) == ["a", "b", "c"]
