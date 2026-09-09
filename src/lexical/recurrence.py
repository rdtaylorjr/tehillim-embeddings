"""Per-half-verse ICF recurrence: mean similarity to others, binned by spacing."""

from __future__ import annotations

import numpy as np

from core.columns import PsalmColumns
from core.similarity import full_cosine_similarity_matrix, half_verse_vector, lag_bin_index
from core.vocabulary import index_map
from lexical.vectorize import icf_vector


def spacing_profile_vectors(
    columns: list[PsalmColumns],
    vocabulary: tuple[str, ...],
    icf_weights: dict[str, float],
    k: int,
    order_by_psalm: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Per-half-verse [r_1,...,r_k]: mean ICF-weighted similarity to every other, by lag bin."""
    weights = icf_vector(vocabulary, icf_weights)
    index_of = index_map(vocabulary)
    dim = len(vocabulary)

    vectors: dict[int, np.ndarray] = {}
    for psalm in columns:
        half_verses = psalm.half_verses
        n = len(half_verses)
        order = order_by_psalm[psalm.number] if order_by_psalm is not None else np.arange(n)

        if n < 2:
            for node in psalm.nodes:
                vectors[node] = np.zeros(k, dtype=np.float32)
            continue

        ordered = [half_verses[i] for i in order]
        half_verse_vectors = np.stack(
            [half_verse_vector(hv, index_of, weights, dim) for hv in ordered]
        )
        similarity = full_cosine_similarity_matrix(half_verse_vectors)
        positions = np.arange(n)

        for position, half_verse_index in enumerate(order):
            others = positions != position
            delta = np.abs(positions[others] - position) / (n - 1)
            bins = lag_bin_index(delta, k)
            sims = similarity[position, others]
            counts = np.bincount(bins, minlength=k)
            sums = np.bincount(bins, weights=sims, minlength=k)
            profile = np.zeros(k, dtype=np.float64)
            nonzero = counts > 0
            profile[nonzero] = sums[nonzero] / counts[nonzero]
            vectors[psalm.nodes[half_verse_index]] = profile.astype(np.float32)
    return vectors
