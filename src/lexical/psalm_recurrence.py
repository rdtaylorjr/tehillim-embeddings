"""Psalm-level ICF lexical-recurrence spacing profile, broadcast to every colon in the psalm."""

from __future__ import annotations

import numpy as np

from lexical.corpus import LexicalPsalm
from lexical.vectorize import icf_vector
from lexical.vocabulary import VocabularyKey, half_verses_for_key


def normalized_lag(n: int) -> np.ndarray:
    """delta_ij = |i-j| / (n-1) for every i<j pair among n cola, in triu_indices(n, k=1) order."""
    rows, cols = np.triu_indices(n, k=1)
    return np.asarray(np.abs(rows - cols) / (n - 1))


def lag_bin_index(delta: np.ndarray, k: int) -> np.ndarray:
    """Which of k equal-width [0, 1] lag-distance bins each normalized separation falls into."""
    return np.asarray(np.minimum((delta * k).astype(int), k - 1))


def _colon_vector(
    half_verse: tuple[str, ...], index_of: dict[str, int], weights: np.ndarray, dim: int
) -> np.ndarray:
    indices = np.fromiter((index_of[v] for v in set(half_verse) if v in index_of), dtype=np.int64)
    vector = np.zeros(dim, dtype=np.float32)
    vector[indices] = weights[indices]
    return vector


def _pairwise_cosine_similarity(vectors: np.ndarray) -> np.ndarray:
    """cos(x_i, x_j) for every i<j pair, in triu_indices order; 0 for a zero-norm colon vector."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    normalized = vectors / safe_norms
    similarity = normalized @ normalized.T
    rows, cols = np.triu_indices(len(vectors), k=1)
    return np.asarray(similarity[rows, cols])


def psalm_spacing_profile_vectors(
    psalms: list[LexicalPsalm],
    vocabulary: tuple[str, ...],
    key: VocabularyKey,
    icf_weights: dict[str, float],
    k: int,
    order_by_psalm: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Psalm-level [r_1,...,r_k]: mean ICF-weighted colon-pair cosine similarity per lag bin."""
    weights = icf_vector(vocabulary, icf_weights)
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)

    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        half_verses = half_verses_for_key(psalm, key)
        n = len(half_verses)
        order = order_by_psalm[psalm.number] if order_by_psalm is not None else np.arange(n)
        ordered = [half_verses[i] for i in order]

        profile = np.zeros(k, dtype=np.float32)
        if n >= 2:
            colon_vectors = np.stack([_colon_vector(hv, index_of, weights, dim) for hv in ordered])
            delta = normalized_lag(n)
            bins = lag_bin_index(delta, k)
            similarities = _pairwise_cosine_similarity(colon_vectors)
            counts = np.bincount(bins, minlength=k)
            sums = np.bincount(bins, weights=similarities, minlength=k)
            nonzero = counts > 0
            profile = np.zeros(k, dtype=np.float64)
            profile[nonzero] = sums[nonzero] / counts[nonzero]
            profile = profile.astype(np.float32)

        for node in psalm.half_verse_nodes:
            vectors[node] = profile
    return vectors
