"""Shared clause n-gram vectorizing: rare-collapsed colon sequences into dense or sparse blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from core.columns import PsalmColumns
from core.ngram import (
    bigram_histogram,
    pooled_ngram_psalm_vectors,
    reorder,
    sparse_1_2_3gram,
    sparse_pooled_1_2_3gram,
    trigram_histogram,
    unigram_histogram,
)
from core.support import collapse_rare
from core.vocabulary import index_map
from syntactic.assignment import Assignment, majority_mask
from syntactic.clause_columns import colon_sequences

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from syntactic.corpus import ClausePsalm

__all__ = [
    "ColumnsOf",
    "collapsed_columns",
    "dense_ngram_psalm_vectors",
    "dense_ngram_vectors",
    "sparse_1_2_3gram_psalm_vectors",
    "sparse_1_2_3gram_vectors",
]

type ColumnsOf = Callable[["ClausePsalm"], tuple[tuple[str, ...], ...]]

_HISTOGRAMS = {1: unigram_histogram, 2: bigram_histogram, 3: trigram_histogram}


def collapsed_columns(
    values: Sequence[str],
    assignment: Assignment,
    n_colons: int,
    external_counts: dict[str, int],
    k: int,
    keep: np.ndarray | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Per-colon value sequences under the majority rule, rare values collapsed first."""
    mask = majority_mask(assignment)
    if keep is not None:
        #: Excluded nodes leave through the mask, so the surviving indices still address `values`.
        mask &= keep[assignment.node_index]
    collapsed = tuple(collapse_rare(value, external_counts, k) for value in values)
    return colon_sequences(collapsed, assignment, n_colons, mask=mask)


def dense_ngram_vectors(
    psalms: list[ClausePsalm],
    columns_of: ColumnsOf,
    vocabulary: tuple[str, ...],
    orders: tuple[int, ...],
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """One concatenated n-gram histogram per colon node, over the given orders."""
    index_of = index_map(vocabulary)
    dim = len(vocabulary)
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, column in zip(psalm.half_verse_nodes, columns_of(psalm), strict=True):
            ordered = reorder(column, node, order_by_node)
            blocks = [_HISTOGRAMS[order](ordered, index_of, dim) for order in orders]
            vectors[node] = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]
    return vectors


def dense_ngram_psalm_vectors(
    psalms: list[ClausePsalm],
    columns_of: ColumnsOf,
    vocabulary: tuple[str, ...],
    orders: tuple[int, ...],
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Psalm-broadcast n-grams: raw counts summed across the psalm, normalized once per order."""
    columns = [
        PsalmColumns(psalm.number, psalm.half_verse_nodes, columns_of(psalm)) for psalm in psalms
    ]
    return pooled_ngram_psalm_vectors(columns, orders, vocabulary, order_by_node=order_by_node)


def sparse_1_2_3gram_vectors(
    psalms: list[ClausePsalm],
    columns_of: ColumnsOf,
    vocabulary: tuple[str, ...],
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Sparse `[unigram; bigram; trigram]` per colon node, never materializing the dense width."""
    index_of = index_map(vocabulary)
    dim = len(vocabulary)
    vectors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for psalm in psalms:
        for node, column in zip(psalm.half_verse_nodes, columns_of(psalm), strict=True):
            vectors[node] = sparse_1_2_3gram(reorder(column, node, order_by_node), index_of, dim)
    return vectors


def sparse_1_2_3gram_psalm_vectors(
    psalms: list[ClausePsalm],
    columns_of: ColumnsOf,
    vocabulary: tuple[str, ...],
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Psalm-broadcast sparse `[unigram; bigram; trigram]`, pooled across the psalm's colons."""
    columns = [
        PsalmColumns(psalm.number, psalm.half_verse_nodes, columns_of(psalm)) for psalm in psalms
    ]
    return sparse_pooled_1_2_3gram(columns, vocabulary, order_by_node=order_by_node)
