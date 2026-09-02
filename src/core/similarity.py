"""Lag binning and cosine similarity over per-half-verse vectors, shared by recurrence."""

from __future__ import annotations

import numpy as np


def normalized_lag(n: int) -> np.ndarray:
    """delta_ij = |i-j| / (n-1) for every i<j pair among n, in triu_indices order."""
    rows, cols = np.triu_indices(n, k=1)
    return np.asarray(np.abs(rows - cols) / (n - 1))


def lag_bin_index(delta: np.ndarray, k: int) -> np.ndarray:
    """Which of k equal-width [0, 1] lag-distance bins each normalized separation falls into."""
    return np.asarray(np.minimum((delta * k).astype(int), k - 1))


def half_verse_vector(
    half_verse: tuple[str, ...], index_of: dict[str, int], weights: np.ndarray, dim: int
) -> np.ndarray:
    """One ICF-weighted presence vector over the vocabulary."""
    indices = np.fromiter((index_of[v] for v in set(half_verse) if v in index_of), dtype=np.int64)
    vector = np.zeros(dim, dtype=np.float32)
    vector[indices] = weights[indices]
    return vector


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    """Row-normalized copy, leaving zero-norm rows as zero so their similarity is 0."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


def full_cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """cos(x_i, x_j) for every (i, j) pair, full symmetric matrix; 0 for a zero-norm vector."""
    normalized = _unit_rows(vectors)
    return np.asarray(normalized @ normalized.T)


def pairwise_cosine_similarity(vectors: np.ndarray) -> np.ndarray:
    """cos(x_i, x_j) for every i<j pair, in triu_indices order; 0 for a zero-norm vector."""
    rows, cols = np.triu_indices(len(vectors), k=1)
    return np.asarray(full_cosine_similarity_matrix(vectors)[rows, cols])
