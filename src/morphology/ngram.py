"""Vocabulary-agnostic n-gram counting and histograms, shared by every representation family."""

from __future__ import annotations

import numpy as np

Colon = tuple[str, ...]
PsalmColumns = tuple[tuple[int, ...], tuple[Colon, ...]]


def reorder(values: Colon, node: int, order_by_node: dict[int, np.ndarray] | None) -> Colon:
    """Applies the word-index permutation for `node`, if any, else returns `values` unchanged."""
    if order_by_node is None:
        return values
    order = order_by_node.get(node)
    if order is None:
        return values
    return tuple(values[i] for i in order)


def _indices(values: Colon, index_of: dict[str, int]) -> np.ndarray:
    """Maps each value to its vocabulary index; the only per-word Python-level step."""
    return np.fromiter((index_of[value] for value in values), dtype=np.int64, count=len(values))


def unigram_counts(values: Colon, index_of: dict[str, int], dim: int) -> np.ndarray:
    """Raw per-value occurrence counts over `values`, via a single batched bincount."""
    if not values:
        return np.zeros(dim, dtype=np.float64)
    return np.bincount(_indices(values, index_of), minlength=dim).astype(np.float64)


def bigram_counts(values: Colon, index_of: dict[str, int], dim: int) -> np.ndarray:
    """Raw adjacent-pair occurrence counts over `values`, flattened row-major, batched."""
    if len(values) < 2:
        return np.zeros(dim * dim, dtype=np.float64)
    indices = _indices(values, index_of)
    flat = indices[:-1] * dim + indices[1:]
    return np.bincount(flat, minlength=dim * dim).astype(np.float64)


def trigram_counts(values: Colon, index_of: dict[str, int], dim: int) -> np.ndarray:
    """Raw adjacent-triple occurrence counts over `values`, flattened row-major, batched."""
    if len(values) < 3:
        return np.zeros(dim * dim * dim, dtype=np.float64)
    indices = _indices(values, index_of)
    flat = (indices[:-2] * dim + indices[1:-1]) * dim + indices[2:]
    return np.bincount(flat, minlength=dim * dim * dim).astype(np.float64)


_COUNTERS = {1: unigram_counts, 2: bigram_counts, 3: trigram_counts}


def unigram_histogram(values: Colon, index_of: dict[str, int], dim: int) -> np.ndarray:
    """Normalized value proportions over one colon: count(v) / m."""
    m = len(values)
    counts = unigram_counts(values, index_of, dim)
    return (counts / m if m > 0 else counts).astype(np.float32)


def bigram_histogram(values: Colon, index_of: dict[str, int], dim: int) -> np.ndarray:
    """Normalized adjacent-pair proportions over one colon: count(pair) / (m - 1)."""
    denom = len(values) - 1
    counts = bigram_counts(values, index_of, dim)
    return (counts / denom if denom > 0 else counts).astype(np.float32)


def trigram_histogram(values: Colon, index_of: dict[str, int], dim: int) -> np.ndarray:
    """Normalized adjacent-triple proportions over one colon: count(triple) / (m - 2)."""
    denom = len(values) - 2
    counts = trigram_counts(values, index_of, dim)
    return (counts / denom if denom > 0 else counts).astype(np.float32)


def _sparse_order_counts(flat: np.ndarray, denom: int) -> tuple[np.ndarray, np.ndarray]:
    """Nonzero (index, normalized-count) pairs from realized flat indices, never a dense array."""
    if flat.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    unique_idx, counts = np.unique(flat, return_counts=True)
    return unique_idx, counts / denom


def sparse_1_2_3gram(
    values: Colon, index_of: dict[str, int], dim: int
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
    index_of: dict[str, int],
    dim: int,
    order_by_node: dict[int, np.ndarray] | None,
) -> dict[int, np.ndarray]:
    """Word-count-weighted psalm-wide pooling: sums raw n-gram counts, normalizes once per order."""
    vectors: dict[int, np.ndarray] = {}
    for nodes, colons in psalm_columns:
        totals = {order: np.zeros(dim**order, dtype=np.float64) for order in orders}
        denominators = dict.fromkeys(orders, 0)
        for node, colon_values in zip(nodes, colons, strict=True):
            ordered = reorder(colon_values, node, order_by_node)
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
    index_of: dict[str, int],
    dim: int,
    order_by_node: dict[int, np.ndarray] | None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Word-count-weighted psalm-wide sparse pooling of [unigram; bigram; trigram], never dense."""
    vectors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for nodes, colons in psalm_columns:
        uni_parts, bi_parts, tri_parts = [], [], []
        uni_denom = bi_denom = tri_denom = 0
        for node, colon_values in zip(nodes, colons, strict=True):
            ordered = reorder(colon_values, node, order_by_node)
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
