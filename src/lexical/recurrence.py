"""Per-colon ICF lexical-recurrence: mean similarity to other colons, binned by spacing."""

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
    vector = np.zeros(dim, dtype=np.float32)
    for value in set(half_verse):
        index = index_of.get(value)
        if index is not None:
            vector[index] = weights[index]
    return vector


def _full_cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """cos(x_i, x_j) for every (i, j) pair, full symmetric matrix; 0 for a zero-norm colon."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    normalized = vectors / safe_norms
    return np.asarray(normalized @ normalized.T)


def spacing_profile_vectors(
    psalms: list[LexicalPsalm],
    vocabulary: tuple[str, ...],
    key: VocabularyKey,
    icf_weights: dict[str, float],
    k: int,
    order_by_psalm: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Per-colon [r_1,...,r_k]: mean ICF-weighted similarity to every other colon, by lag bin."""
    weights = icf_vector(vocabulary, icf_weights)
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)

    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        half_verses = half_verses_for_key(psalm, key)
        n = len(half_verses)
        order = order_by_psalm[psalm.number] if order_by_psalm is not None else np.arange(n)

        if n < 2:
            for node in psalm.half_verse_nodes:
                vectors[node] = np.zeros(k, dtype=np.float32)
            continue

        ordered = [half_verses[i] for i in order]
        colon_vectors = np.stack([_colon_vector(hv, index_of, weights, dim) for hv in ordered])
        similarity = _full_cosine_similarity_matrix(colon_vectors)
        positions = np.arange(n)

        for position, colon_index in enumerate(order):
            others = positions != position
            delta = np.abs(positions[others] - position) / (n - 1)
            bins = lag_bin_index(delta, k)
            sims = similarity[position, others]
            profile = np.zeros(k, dtype=np.float32)
            for b in range(k):
                mask = bins == b
                if mask.any():
                    profile[b] = sims[mask].mean()
            vectors[psalm.half_verse_nodes[colon_index]] = profile
    return vectors
