"""Psalm-level ICF lexical-recurrence over surface forms, broadcast to every colon."""

from __future__ import annotations

import numpy as np

from lexical.recurrence import lag_bin_index, normalized_lag
from lexical.surface_corpus import SurfacePsalm
from lexical.surface_vectorize import surface_icf_vector
from lexical.surface_vocabulary import SurfaceTier, half_verses_for_tier


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


def surface_psalm_spacing_profile_vectors(
    psalms: list[SurfacePsalm],
    vocabulary: tuple[str, ...],
    tier: SurfaceTier,
    icf_weights: dict[str, float],
    k: int,
    order_by_psalm: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Psalm-level [r_1,...,r_k]: mean ICF-weighted colon-pair cosine similarity per lag bin."""
    weights = surface_icf_vector(vocabulary, icf_weights)
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)

    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        half_verses = half_verses_for_tier(psalm, tier)
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
