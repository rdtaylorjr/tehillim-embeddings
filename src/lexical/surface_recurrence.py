"""Per-colon ICF lexical-recurrence over surface forms: similarity to other colons, by spacing."""

from __future__ import annotations

import numpy as np

from lexical.recurrence import lag_bin_index
from lexical.surface_corpus import SurfacePsalm
from lexical.surface_vectorize import surface_icf_vector
from lexical.surface_vocabulary import SurfaceTier, cola_for_tier


def _colon_vector(
    colon: tuple[str, ...], index_of: dict[str, int], weights: np.ndarray, dim: int
) -> np.ndarray:
    indices = np.fromiter((index_of[v] for v in set(colon) if v in index_of), dtype=np.int64)
    vector = np.zeros(dim, dtype=np.float32)
    vector[indices] = weights[indices]
    return vector


def _full_cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """cos(x_i, x_j) for every (i, j) pair, full symmetric matrix; 0 for a zero-norm colon."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    normalized = vectors / safe_norms
    return np.asarray(normalized @ normalized.T)


def surface_spacing_profile_vectors(
    psalms: list[SurfacePsalm],
    vocabulary: tuple[str, ...],
    tier: SurfaceTier,
    icf_weights: dict[str, float],
    k: int,
    order_by_psalm: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Per-colon [r_1,...,r_k]: mean ICF-weighted similarity to every other colon, by lag bin."""
    weights = surface_icf_vector(vocabulary, icf_weights)
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)

    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        cola = cola_for_tier(psalm, tier)
        n = len(cola)
        order = order_by_psalm[psalm.number] if order_by_psalm is not None else np.arange(n)

        if n < 2:
            for node in psalm.colon_nodes:
                vectors[node] = np.zeros(k, dtype=np.float32)
            continue

        ordered = [cola[i] for i in order]
        colon_vectors = np.stack([_colon_vector(hv, index_of, weights, dim) for hv in ordered])
        similarity = _full_cosine_similarity_matrix(colon_vectors)
        positions = np.arange(n)

        for position, colon_index in enumerate(order):
            others = positions != position
            delta = np.abs(positions[others] - position) / (n - 1)
            bins = lag_bin_index(delta, k)
            sims = similarity[position, others]
            counts = np.bincount(bins, minlength=k)
            sums = np.bincount(bins, weights=sims, minlength=k)
            profile = np.zeros(k, dtype=np.float64)
            nonzero = counts > 0
            profile[nonzero] = sums[nonzero] / counts[nonzero]
            vectors[psalm.colon_nodes[colon_index]] = profile.astype(np.float32)
    return vectors
