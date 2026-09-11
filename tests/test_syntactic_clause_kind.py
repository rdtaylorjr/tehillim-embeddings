from __future__ import annotations

import numpy as np
import pytest

from syntactic.assignment import Assignment
from syntactic.clause_kind import (
    clause_kind_1gram_psalm_vectors,
    clause_kind_1gram_vectors,
    clause_kind_columns,
)
from syntactic.clause_vocabulary import KIND_VOCABULARY
from syntactic.corpus import ClausePsalm

NC, VC, WP = (KIND_VOCABULARY.index(v) for v in ("NC", "VC", "WP"))


def _psalm(number, nodes, kinds, node_index, unit_index, weight):
    return ClausePsalm(
        number=number,
        half_verse_nodes=nodes,
        clause_nodes=tuple(range(len(kinds))),
        clause_kind=kinds,
        clause_assignment=Assignment(
            node_index=np.array(node_index),
            unit_index=np.array(unit_index),
            weight=np.array(weight),
        ),
    )


class TestClauseKindColumns:
    def test_one_sequence_per_colon_in_node_order(self):
        psalm = _psalm(1, (10, 11), ("VC", "NC", "WP"), [0, 1, 2], [0, 0, 1], [1.0, 1.0, 1.0])

        assert clause_kind_columns(psalm) == (("VC", "NC"), ("WP",))


class TestClauseKindInventoryVectors:
    def test_a_colon_histogram_is_the_proportion_of_each_kind(self):
        psalm = _psalm(1, (10,), ("VC", "VC", "NC"), [0, 1, 2], [0, 0, 0], [1.0, 1.0, 1.0])

        vector = clause_kind_1gram_vectors([psalm])[10]

        assert vector[VC] == pytest.approx(2 / 3)
        assert vector[NC] == pytest.approx(1 / 3)
        assert vector[WP] == pytest.approx(0.0)

    def test_a_histogram_sums_to_one_when_the_colon_has_a_clause(self):
        psalm = _psalm(1, (10,), ("WP",), [0], [0], [1.0])

        assert clause_kind_1gram_vectors([psalm])[10].sum() == pytest.approx(1.0)

    def test_a_colon_no_clause_reaches_is_the_zero_vector(self):
        psalm = _psalm(1, (10, 11), ("VC",), [0], [0], [1.0])

        assert clause_kind_1gram_vectors([psalm])[11].tolist() == [0.0, 0.0, 0.0]

    def test_the_vector_width_is_the_three_bhsa_classes(self):
        psalm = _psalm(1, (10,), ("VC",), [0], [0], [1.0])

        assert clause_kind_1gram_vectors([psalm])[10].shape == (3,)

    def test_every_colon_node_gets_a_vector(self):
        psalm = _psalm(1, (10, 11, 12), ("VC", "NC"), [0, 1], [0, 2], [1.0, 1.0])

        assert set(clause_kind_1gram_vectors([psalm])) == {10, 11, 12}


class TestClauseKindInventoryPsalmVectors:
    def test_every_colon_of_a_psalm_shares_one_pooled_vector(self):
        psalm = _psalm(1, (10, 11), ("VC", "NC"), [0, 1], [0, 1], [1.0, 1.0])

        vectors = clause_kind_1gram_psalm_vectors([psalm])

        assert vectors[10].tolist() == vectors[11].tolist()

    def test_pooling_counts_every_clause_in_the_psalm_not_just_one_colon(self):
        psalm = _psalm(1, (10, 11), ("VC", "VC", "NC"), [0, 1, 2], [0, 0, 1], [1.0, 1.0, 1.0])

        vector = clause_kind_1gram_psalm_vectors([psalm])[10]

        assert vector[VC] == pytest.approx(2 / 3)
        assert vector[NC] == pytest.approx(1 / 3)
