from __future__ import annotations

import numpy as np
import pytest

from syntactic.assignment import Assignment
from syntactic.clause_edge import (
    EDGE_DISTANCE_VOCABULARY,
    NO_MOTHER,
    clause_edge_distance_columns,
    distance_bin,
    edge_reaches_outside_colon,
)
from syntactic.corpus import ClausePsalm


def _psalm(nodes, mother, node_index, unit_index, weight):
    n = len(mother)
    return ClausePsalm(
        number=1,
        half_verse_nodes=nodes,
        clause_atom_nodes=tuple(range(n)),
        clause_atom_typ=("NmCl",) * n,
        clause_atom_mother=mother,
        clause_atom_assignment=Assignment(
            node_index=np.array(node_index),
            unit_index=np.array(unit_index),
            weight=np.array(weight),
        ),
    )


class TestDistanceBin:
    @pytest.mark.parametrize(
        ("distance", "expected"),
        [(1, "1"), (-1, "1"), (2, "2"), (-2, "2"), (3, "3-4"), (4, "3-4"), (5, "5+"), (-9, "5+")],
    )
    def test_bins_by_absolute_span(self, distance, expected):
        assert distance_bin(distance) == expected

    def test_the_vocabulary_carries_no_zero_because_a_self_mother_cannot_occur(self):
        assert "0" not in EDGE_DISTANCE_VOCABULARY

    def test_the_vocabulary_is_closed_and_structural_so_no_threshold_applies(self):
        """Four spans and a root marker, unlike the open lexical vocabularies."""
        assert EDGE_DISTANCE_VOCABULARY == ("1", "2", "3-4", "5+", NO_MOTHER)

    def test_every_span_bin_is_reachable(self):
        assert {distance_bin(d) for d in (1, 2, 3, 4, 5, 100)} == {"1", "2", "3-4", "5+"}


class TestClauseEdgeDistanceColumns:
    def test_a_rootless_atom_is_marked_rather_than_dropped(self):
        """A root carries no edge, and omitting it would shrink the colon's mass."""
        psalm = _psalm((10,), (None,), [0], [0], [1.0])

        assert clause_edge_distance_columns(psalm) == ((NO_MOTHER,),)

    def test_an_edge_is_keyed_to_its_daughters_colon(self):
        psalm = _psalm((10, 11), (None, 0), [0, 1], [0, 1], [1.0, 1.0])

        assert clause_edge_distance_columns(psalm) == ((NO_MOTHER,), ("1",))

    def test_a_mother_outside_the_extracted_atoms_counts_as_rootless(self):
        psalm = _psalm((10,), (999,), [0], [0], [1.0])

        assert clause_edge_distance_columns(psalm) == ((NO_MOTHER,),)

    def test_a_long_span_reaches_the_top_bin(self):
        mother = (None, None, None, None, None, None, 0)
        psalm = _psalm((10,), mother, list(range(7)), [0] * 7, [1.0] * 7)

        assert clause_edge_distance_columns(psalm)[0][-1] == "5+"


class TestEdgeReachesOutsideColon:
    def test_flags_a_daughter_whose_mother_sits_in_another_colon(self):
        psalm = _psalm((10, 11), (None, 0), [0, 1], [0, 1], [1.0, 1.0])

        assert edge_reaches_outside_colon(psalm).tolist() == [False, True]

    def test_a_same_colon_edge_is_not_flagged(self):
        psalm = _psalm((10,), (None, 0), [0, 1], [0, 0], [1.0, 1.0])

        assert edge_reaches_outside_colon(psalm).tolist() == [False, False]

    def test_a_root_is_never_flagged(self):
        psalm = _psalm((10,), (None,), [0], [0], [1.0])

        assert edge_reaches_outside_colon(psalm).tolist() == [False]
