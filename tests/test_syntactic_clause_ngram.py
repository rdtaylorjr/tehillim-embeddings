from __future__ import annotations

import numpy as np
import pytest

from core.support import RARE_TOKEN
from syntactic.assignment import Assignment
from syntactic.clause_ngram import (
    collapsed_columns,
    dense_ngram_psalm_vectors,
    dense_ngram_vectors,
    sparse_1_2_3gram_vectors,
)
from syntactic.corpus import ClausePsalm

VOCAB = ("A", "B", RARE_TOKEN)
COUNTS = {"A": 5000, "B": 5000, "C": 10}
K = 1000


def _assignment(node_index, unit_index, weight):
    return Assignment(
        node_index=np.array(node_index),
        unit_index=np.array(unit_index),
        weight=np.array(weight),
    )


def _psalm(nodes, typ, node_index, unit_index, weight):
    return ClausePsalm(
        number=1,
        half_verse_nodes=nodes,
        clause_atom_nodes=tuple(range(len(typ))),
        clause_atom_typ=typ,
        clause_atom_assignment=_assignment(node_index, unit_index, weight),
    )


def _columns(psalm):
    return collapsed_columns(
        psalm.clause_atom_typ,
        psalm.clause_atom_assignment,
        len(psalm.half_verse_nodes),
        COUNTS,
        K,
    )


class TestCollapsedColumns:
    def test_a_value_below_the_threshold_becomes_the_rare_token(self):
        psalm = _psalm((10,), ("A", "C"), [0, 1], [0, 0], [1.0, 1.0])

        assert _columns(psalm) == (("A", RARE_TOKEN),)

    def test_a_value_absent_from_the_support_table_is_also_rare(self):
        psalm = _psalm((10,), ("Z",), [0], [0], [1.0])

        assert _columns(psalm) == ((RARE_TOKEN,),)

    def test_collapsing_happens_before_grouping_so_order_is_preserved(self):
        psalm = _psalm((10, 11), ("A", "C", "B"), [0, 1, 2], [0, 1, 1], [1.0, 1.0, 1.0])

        assert _columns(psalm) == (("A",), (RARE_TOKEN, "B"))


class TestDenseNgramVectors:
    def test_a_unigram_vector_is_the_value_proportions(self):
        psalm = _psalm((10,), ("A", "A", "B"), [0, 1, 2], [0, 0, 0], [1.0, 1.0, 1.0])

        vector = dense_ngram_vectors([psalm], _columns, VOCAB, (1,))[10]

        assert vector.tolist() == pytest.approx([2 / 3, 1 / 3, 0.0])

    def test_concatenating_two_orders_stacks_their_blocks(self):
        psalm = _psalm((10,), ("A", "B"), [0, 1], [0, 0], [1.0, 1.0])

        vector = dense_ngram_vectors([psalm], _columns, VOCAB, (1, 2))[10]

        assert vector.shape == (len(VOCAB) + len(VOCAB) ** 2,)
        assert vector[: len(VOCAB)].tolist() == pytest.approx([0.5, 0.5, 0.0])

    def test_a_bigram_block_records_the_adjacent_pair(self):
        psalm = _psalm((10,), ("A", "B"), [0, 1], [0, 0], [1.0, 1.0])

        vector = dense_ngram_vectors([psalm], _columns, VOCAB, (2,))[10]

        a_then_b = VOCAB.index("A") * len(VOCAB) + VOCAB.index("B")
        assert vector[a_then_b] == pytest.approx(1.0)

    def test_a_colon_with_one_clause_has_no_bigram(self):
        psalm = _psalm((10,), ("A",), [0], [0], [1.0])

        assert dense_ngram_vectors([psalm], _columns, VOCAB, (2,))[10].sum() == pytest.approx(0.0)

    def test_every_colon_node_gets_a_vector(self):
        psalm = _psalm((10, 11), ("A",), [0], [0], [1.0])

        assert set(dense_ngram_vectors([psalm], _columns, VOCAB, (1,))) == {10, 11}


class TestDenseNgramPsalmVectors:
    def test_pooling_spans_the_psalms_colons(self):
        psalm = _psalm((10, 11), ("A", "A", "B"), [0, 1, 2], [0, 0, 1], [1.0, 1.0, 1.0])

        vectors = dense_ngram_psalm_vectors([psalm], _columns, VOCAB, (1,))

        assert vectors[10].tolist() == vectors[11].tolist()
        assert vectors[10].tolist() == pytest.approx([2 / 3, 1 / 3, 0.0])


class TestSparseVectors:
    def test_a_sparse_vector_carries_only_realized_indices(self):
        psalm = _psalm((10,), ("A", "B"), [0, 1], [0, 0], [1.0, 1.0])

        indices, values = sparse_1_2_3gram_vectors([psalm], _columns, VOCAB)[10]

        assert indices.size == values.size
        assert values.size > 0
        assert np.all(values > 0)

    def test_an_empty_colon_carries_nothing(self):
        psalm = _psalm((10, 11), ("A",), [0], [0], [1.0])

        indices, values = sparse_1_2_3gram_vectors([psalm], _columns, VOCAB)[11]

        assert indices.size == 0
        assert values.size == 0


class TestOrderShuffleContract:
    def test_reordering_leaves_the_unigram_block_untouched(self):
        """The shuffle null must vary order alone, so inventory has to survive it exactly."""
        psalm = _psalm((10,), ("A", "B", "B"), [0, 1, 2], [0, 0, 0], [1.0, 1.0, 1.0])
        order = {10: np.array([2, 0, 1])}

        plain = dense_ngram_vectors([psalm], _columns, VOCAB, (1,))[10]
        shuffled = dense_ngram_vectors([psalm], _columns, VOCAB, (1,), order)[10]

        assert shuffled.tolist() == pytest.approx(plain.tolist())

    def test_reordering_changes_the_bigram_block(self):
        psalm = _psalm((10,), ("A", "B"), [0, 1], [0, 0], [1.0, 1.0])
        order = {10: np.array([1, 0])}

        plain = dense_ngram_vectors([psalm], _columns, VOCAB, (2,))[10]
        shuffled = dense_ngram_vectors([psalm], _columns, VOCAB, (2,), order)[10]

        assert not np.allclose(plain, shuffled)

    def test_a_colon_absent_from_the_order_map_keeps_its_corpus_order(self):
        psalm = _psalm((10,), ("A", "B"), [0, 1], [0, 0], [1.0, 1.0])

        plain = dense_ngram_vectors([psalm], _columns, VOCAB, (2,))[10]
        partial = dense_ngram_vectors([psalm], _columns, VOCAB, (2,), {})[10]

        assert partial.tolist() == pytest.approx(plain.tolist())
