from __future__ import annotations

import numpy as np
import pytest

from syntactic.assignment import Assignment
from syntactic.clause_tree import (
    TREE_SUMMARY_FIELDS,
    clause_tree_summary,
    clause_tree_summary_psalm_vectors,
    mother_positions,
)
from syntactic.corpus import ClausePsalm

FIELD = {name: i for i, name in enumerate(TREE_SUMMARY_FIELDS)}


def _psalm(nodes, mother, n_atoms=None):
    n = n_atoms if n_atoms is not None else len(mother)
    return ClausePsalm(
        number=1,
        half_verse_nodes=nodes,
        clause_atom_nodes=tuple(range(n)),
        clause_atom_typ=("NmCl",) * n,
        clause_atom_mother=mother,
        clause_atom_assignment=Assignment(
            node_index=np.arange(n),
            unit_index=np.zeros(n, dtype=np.int64),
            weight=np.ones(n),
        ),
    )


class TestMotherPositions:
    def test_a_root_is_minus_one(self):
        assert mother_positions(_psalm((10,), (None, 0))).tolist() == [-1, 0]

    def test_a_mother_outside_the_psalm_is_also_minus_one(self):
        assert mother_positions(_psalm((10,), (None, 999))).tolist() == [-1, -1]


class TestClauseTreeSummary:
    def test_a_chain_reports_its_depth(self):
        summary = clause_tree_summary(_psalm((10,), (None, 0, 1, 2)))

        assert summary[FIELD["max_depth"]] == pytest.approx(3.0)
        assert summary[FIELD["mean_depth"]] == pytest.approx(1.5)

    def test_a_flat_forest_has_zero_depth_and_every_node_a_root(self):
        summary = clause_tree_summary(_psalm((10,), (None, None, None)))

        assert summary[FIELD["max_depth"]] == pytest.approx(0.0)
        assert summary[FIELD["n_roots"]] == pytest.approx(3.0)
        assert summary[FIELD["leaf_proportion"]] == pytest.approx(1.0)

    def test_branching_counts_daughters_per_node(self):
        summary = clause_tree_summary(_psalm((10,), (None, 0, 0, 0)))

        assert summary[FIELD["max_branching"]] == pytest.approx(3.0)
        assert summary[FIELD["leaf_proportion"]] == pytest.approx(0.75)

    def test_link_span_proportions_split_adjacent_from_long_range(self):
        #: positions 1 and 5 both depend on 0, spans of 1 and 5
        summary = clause_tree_summary(_psalm((10,), (None, 0, None, None, None, 0)))

        assert summary[FIELD["adjacent_link_proportion"]] == pytest.approx(0.5)
        assert summary[FIELD["long_range_link_proportion"]] == pytest.approx(0.5)

    def test_an_empty_psalm_is_all_zeros_rather_than_an_error(self):
        summary = clause_tree_summary(_psalm((), (), n_atoms=0))

        assert summary.tolist() == [0.0] * len(TREE_SUMMARY_FIELDS)

    def test_the_vector_width_matches_the_declared_fields(self):
        assert clause_tree_summary(_psalm((10,), (None, 0))).shape == (len(TREE_SUMMARY_FIELDS),)

    def test_a_cycle_is_refused_rather_than_looping_forever(self):
        with pytest.raises(ValueError, match="cycle"):
            clause_tree_summary(_psalm((10,), (1, 0)))


class TestPsalmVectors:
    def test_every_colon_of_a_psalm_carries_the_same_forest_shape(self):
        psalm = _psalm((10, 11), (None, 0))

        vectors = clause_tree_summary_psalm_vectors([psalm])

        assert set(vectors) == {10, 11}
        assert vectors[10].tolist() == vectors[11].tolist()
