"""Vocabulary-agnostic n-gram counting and histograms, shared by every representation family."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from core.columns import PsalmColumns
from core.shuffle import NumberedPsalm
from core.support import collapsed_sequences
from core.vocabulary import index_map

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

HalfVerse = tuple[str, ...]

#: Reads one psalm's per-node categorical sequences, whichever feature the caller is vectorizing.
type ColumnsOf[PsalmT] = Callable[[PsalmT], tuple[HalfVerse, ...]]


def reorder(values: HalfVerse, node: int, order_by_node: dict[int, np.ndarray] | None) -> HalfVerse:
    """Applies the word-index permutation for `node`, if any, else returns `values` unchanged."""
    if order_by_node is None:
        return values
    order = order_by_node.get(node)
    if order is None:
        return values
    if len(order) != len(values):
        raise ValueError(
            f"node {node} has a permutation of length {len(order)} "
            f"for a sequence of length {len(values)}"
        )
    return tuple(values[i] for i in order)


def permuted_columns(
    columns: tuple[HalfVerse, ...],
    nodes: tuple[int, ...],
    order_by_node: dict[int, np.ndarray] | None,
) -> tuple[HalfVerse, ...]:
    """Each node's sequence under its permutation, while the sequence is still the real one."""
    if order_by_node is None:
        return columns
    return tuple(
        reorder(column, node, order_by_node) for node, column in zip(nodes, columns, strict=True)
    )


def _indices(values: HalfVerse, index_of: dict[str, int]) -> np.ndarray:
    """Maps each value to its vocabulary index; the only per-word Python-level step."""
    return np.fromiter((index_of[value] for value in values), dtype=np.int64, count=len(values))


def unigram_counts(values: HalfVerse, index_of: dict[str, int], dim: int) -> np.ndarray:
    """Raw per-value occurrence counts over `values`, via a single batched bincount."""
    if not values:
        return np.zeros(dim, dtype=np.float64)
    return np.bincount(_indices(values, index_of), minlength=dim).astype(np.float64)


def bigram_counts(values: HalfVerse, index_of: dict[str, int], dim: int) -> np.ndarray:
    """Raw adjacent-pair occurrence counts over `values`, flattened row-major, batched."""
    if len(values) < 2:
        return np.zeros(dim * dim, dtype=np.float64)
    indices = _indices(values, index_of)
    flat = indices[:-1] * dim + indices[1:]
    return np.bincount(flat, minlength=dim * dim).astype(np.float64)


def trigram_counts(values: HalfVerse, index_of: dict[str, int], dim: int) -> np.ndarray:
    """Raw adjacent-triple occurrence counts over `values`, flattened row-major, batched."""
    if len(values) < 3:
        return np.zeros(dim * dim * dim, dtype=np.float64)
    indices = _indices(values, index_of)
    flat = (indices[:-2] * dim + indices[1:-1]) * dim + indices[2:]
    return np.bincount(flat, minlength=dim * dim * dim).astype(np.float64)


_COUNTERS = {1: unigram_counts, 2: bigram_counts, 3: trigram_counts}


def unigram_histogram(values: HalfVerse, index_of: dict[str, int], dim: int) -> np.ndarray:
    """Normalized value proportions over one half-verse: count(v) / m."""
    m = len(values)
    counts = unigram_counts(values, index_of, dim)
    return (counts / m if m > 0 else counts).astype(np.float32)


def bigram_histogram(values: HalfVerse, index_of: dict[str, int], dim: int) -> np.ndarray:
    """Normalized adjacent-pair proportions over one half-verse: count(pair) / (m - 1)."""
    denom = len(values) - 1
    counts = bigram_counts(values, index_of, dim)
    return (counts / denom if denom > 0 else counts).astype(np.float32)


def trigram_histogram(values: HalfVerse, index_of: dict[str, int], dim: int) -> np.ndarray:
    """Normalized adjacent-triple proportions over one half-verse: count(triple) / (m - 2)."""
    denom = len(values) - 2
    counts = trigram_counts(values, index_of, dim)
    return (counts / denom if denom > 0 else counts).astype(np.float32)


def _sparse_order_counts(flat: np.ndarray, denom: int) -> tuple[np.ndarray, np.ndarray]:
    """Nonzero (index, normalized-count) pairs from realized flat indices, never a dense array."""
    if flat.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    unique_idx, counts = np.unique(flat, return_counts=True)
    return unique_idx, counts / denom


def concatenated_1_2_3gram_dim(dim: int) -> int:
    """Width of the concatenated [unigram; bigram; trigram] vector over a dim-sized vocabulary."""
    return dim + dim * dim + dim * dim * dim


def sparse_1_2_3gram(
    values: HalfVerse, index_of: dict[str, int], dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Nonzero (index, value) pairs of [unigram; bigram; trigram], no dense dim-sized array ever."""
    m = len(values)
    if m == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
    idx = _indices(values, index_of)

    uni_idx, uni_val = _sparse_order_counts(idx, m)
    bi_idx, bi_val = (
        _sparse_order_counts(idx[:-1] * dim + idx[1:], m - 1)
        if m >= 2
        else (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64))
    )
    tri_idx, tri_val = (
        _sparse_order_counts((idx[:-2] * dim + idx[1:-1]) * dim + idx[2:], m - 2)
        if m >= 3
        else (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64))
    )

    combined_idx = np.concatenate([uni_idx, dim + bi_idx, dim + dim * dim + tri_idx])
    combined_val = np.concatenate([uni_val, bi_val, tri_val])
    return combined_idx.astype(np.int32), combined_val.astype(np.float32)


def pooled_ngram_psalm_vectors(
    psalm_columns: list[PsalmColumns],
    orders: tuple[int, ...],
    vocabulary: Sequence[str],
    order_by_node: dict[int, np.ndarray] | None,
) -> dict[int, np.ndarray]:
    """Word-count-weighted psalm-wide pooling: sums raw n-gram counts, normalizes once per order."""
    #: Derived once per dataset here, rather than by every caller keeping the two in step.
    index_of = index_map(vocabulary)
    dim = len(vocabulary)
    vectors: dict[int, np.ndarray] = {}
    for columns in psalm_columns:
        nodes, half_verses = columns.nodes, columns.half_verses
        totals = {order: np.zeros(dim**order, dtype=np.float64) for order in orders}
        denominators = dict.fromkeys(orders, 0)
        for node, half_verse_values in zip(nodes, half_verses, strict=True):
            ordered = reorder(half_verse_values, node, order_by_node)
            m = len(ordered)
            for order in orders:
                totals[order] += _COUNTERS[order](ordered, index_of, dim)
                denominators[order] += max(m - (order - 1), 0)
        blocks = []
        for order in orders:
            denom = denominators[order]
            block = totals[order] / denom if denom > 0 else totals[order]
            blocks.append(block.astype(np.float32))
        psalm_vector = np.concatenate(blocks)
        for node in nodes:
            vectors[node] = psalm_vector
    return vectors


def sparse_pooled_1_2_3gram(
    psalm_columns: list[PsalmColumns],
    vocabulary: Sequence[str],
    order_by_node: dict[int, np.ndarray] | None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Word-count-weighted psalm-wide sparse pooling of [unigram; bigram; trigram], never dense."""
    #: Derived once per dataset here, rather than by every caller keeping the two in step.
    index_of = index_map(vocabulary)
    dim = len(vocabulary)
    vectors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for columns in psalm_columns:
        nodes, half_verses = columns.nodes, columns.half_verses
        uni_parts, bi_parts, tri_parts = [], [], []
        uni_denom = bi_denom = tri_denom = 0
        for node, half_verse_values in zip(nodes, half_verses, strict=True):
            ordered = reorder(half_verse_values, node, order_by_node)
            m = len(ordered)
            if m == 0:
                continue
            idx = _indices(ordered, index_of)
            uni_parts.append(idx)
            uni_denom += m
            if m >= 2:
                bi_parts.append(idx[:-1] * dim + idx[1:])
                bi_denom += m - 1
            if m >= 3:
                tri_parts.append((idx[:-2] * dim + idx[1:-1]) * dim + idx[2:])
                tri_denom += m - 2

        empty = np.zeros(0, dtype=np.int64)
        uni_idx, uni_val = _sparse_order_counts(
            np.concatenate(uni_parts) if uni_parts else empty, uni_denom
        )
        bi_idx, bi_val = _sparse_order_counts(
            np.concatenate(bi_parts) if bi_parts else empty, bi_denom
        )
        tri_idx, tri_val = _sparse_order_counts(
            np.concatenate(tri_parts) if tri_parts else empty, tri_denom
        )

        combined_idx = np.concatenate([uni_idx, dim + bi_idx, dim + dim * dim + tri_idx]).astype(
            np.int32
        )
        combined_val = np.concatenate([uni_val, bi_val, tri_val]).astype(np.float32)
        for node in nodes:
            vectors[node] = (combined_idx, combined_val)
    return vectors


_HISTOGRAMS = {1: unigram_histogram, 2: bigram_histogram, 3: trigram_histogram}


def ngram_vectors[PsalmT: NumberedPsalm](
    psalms: Sequence[PsalmT],
    columns_of: ColumnsOf[PsalmT],
    vocabulary: tuple[str, ...],
    orders: tuple[int, ...],
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """One concatenated n-gram histogram per node, over the given orders."""
    index_of = index_map(vocabulary)
    dim = len(vocabulary)
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, column in zip(psalm.half_verse_nodes, columns_of(psalm), strict=True):
            ordered = reorder(column, node, order_by_node)
            blocks = [_HISTOGRAMS[order](ordered, index_of, dim) for order in orders]
            vectors[node] = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]
    return vectors


def ngram_psalm_vectors[PsalmT: NumberedPsalm](
    psalms: Sequence[PsalmT],
    columns_of: ColumnsOf[PsalmT],
    vocabulary: tuple[str, ...],
    orders: tuple[int, ...],
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Psalm-broadcast n-grams: raw counts summed across the psalm, normalized once per order."""
    return pooled_ngram_psalm_vectors(
        _psalm_columns(psalms, columns_of), orders, vocabulary, order_by_node=order_by_node
    )


def sparse_ngram_vectors[PsalmT: NumberedPsalm](
    psalms: Sequence[PsalmT],
    columns_of: ColumnsOf[PsalmT],
    vocabulary: tuple[str, ...],
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Sparse `[unigram; bigram; trigram]` per node, never materializing the dense width."""
    index_of = index_map(vocabulary)
    dim = len(vocabulary)
    vectors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for psalm in psalms:
        for node, column in zip(psalm.half_verse_nodes, columns_of(psalm), strict=True):
            vectors[node] = sparse_1_2_3gram(reorder(column, node, order_by_node), index_of, dim)
    return vectors


def sparse_ngram_psalm_vectors[PsalmT: NumberedPsalm](
    psalms: Sequence[PsalmT],
    columns_of: ColumnsOf[PsalmT],
    vocabulary: tuple[str, ...],
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Psalm-broadcast sparse `[unigram; bigram; trigram]`, pooled across the psalm's nodes."""
    return sparse_pooled_1_2_3gram(
        _psalm_columns(psalms, columns_of), vocabulary, order_by_node=order_by_node
    )


def _psalm_columns[PsalmT: NumberedPsalm](
    psalms: Sequence[PsalmT], columns_of: ColumnsOf[PsalmT]
) -> list[PsalmColumns]:
    """The column view the psalm-pooled builders read, one record per psalm."""
    return [
        PsalmColumns(psalm.number, psalm.half_verse_nodes, columns_of(psalm)) for psalm in psalms
    ]


def _collapsed_columns_of[PsalmT](
    signatures_of: Callable[[PsalmT], tuple[HalfVerse, ...]],
    external_counts: dict[str, int],
    k: int,
    psalm: PsalmT,
) -> tuple[HalfVerse, ...]:
    """One psalm's signature sequences, read then RARE-collapsed against the support table."""
    return collapsed_sequences(signatures_of(psalm), external_counts, k)


def signature_vectors[PsalmT: NumberedPsalm](
    signatures_of: Callable[[PsalmT], tuple[HalfVerse, ...]],
    orders: tuple[int, ...],
    psalms: Sequence[PsalmT],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """N-grams over signature sequences, sub-threshold values collapsed to RARE first."""
    columns_of = partial(_collapsed_columns_of, signatures_of, external_counts, k)
    return ngram_vectors(psalms, columns_of, vocabulary, orders, order_by_node)


def signature_psalm_vectors[PsalmT: NumberedPsalm](
    signatures_of: Callable[[PsalmT], tuple[HalfVerse, ...]],
    orders: tuple[int, ...],
    psalms: Sequence[PsalmT],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Psalm-broadcast n-grams over RARE-collapsed signature sequences."""
    columns_of = partial(_collapsed_columns_of, signatures_of, external_counts, k)
    return ngram_psalm_vectors(psalms, columns_of, vocabulary, orders, order_by_node)


def sparse_signature_vectors[PsalmT: NumberedPsalm](
    signatures_of: Callable[[PsalmT], tuple[HalfVerse, ...]],
    psalms: Sequence[PsalmT],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Sparse trigram concatenation over RARE-collapsed signature sequences."""
    columns_of = partial(_collapsed_columns_of, signatures_of, external_counts, k)
    return sparse_ngram_vectors(psalms, columns_of, vocabulary, order_by_node)


def sparse_signature_psalm_vectors[PsalmT: NumberedPsalm](
    signatures_of: Callable[[PsalmT], tuple[HalfVerse, ...]],
    psalms: Sequence[PsalmT],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Psalm-broadcast sparse trigram concatenation over RARE-collapsed signature sequences."""
    columns_of = partial(_collapsed_columns_of, signatures_of, external_counts, k)
    return sparse_ngram_psalm_vectors(psalms, columns_of, vocabulary, order_by_node)
