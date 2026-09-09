"""Psalm-level ICF lexical-recurrence: mean half-verse-pair similarity, binned by spacing."""

from __future__ import annotations

import numpy as np

from core.columns import PsalmColumns
from core.similarity import (
    half_verse_vector,
    lag_bin_index,
    normalized_lag,
    pairwise_cosine_similarity,
)
from core.vocabulary import index_map
from lexical.vectorize import icf_vector


def psalm_spacing_profile_vectors(
    columns: list[PsalmColumns],
    vocabulary: tuple[str, ...],
    icf_weights: dict[str, float],
    k: int,
    order_by_psalm: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Psalm-level [r_1,...,r_k]: mean ICF-weighted pair cosine similarity per lag bin."""
    weights = icf_vector(vocabulary, icf_weights)
    index_of = index_map(vocabulary)
    dim = len(vocabulary)

    vectors: dict[int, np.ndarray] = {}
    for psalm in columns:
        half_verses = psalm.half_verses
        n = len(half_verses)
        order = order_by_psalm[psalm.number] if order_by_psalm is not None else np.arange(n)
        ordered = [half_verses[i] for i in order]

        profile = np.zeros(k, dtype=np.float32)
        if n >= 2:
            half_verse_vectors = np.stack(
                [half_verse_vector(hv, index_of, weights, dim) for hv in ordered]
            )
            delta = normalized_lag(n)
            bins = lag_bin_index(delta, k)
            similarities = pairwise_cosine_similarity(half_verse_vectors)
            counts = np.bincount(bins, minlength=k)
            sums = np.bincount(bins, weights=similarities, minlength=k)
            nonzero = counts > 0
            profile = np.zeros(k, dtype=np.float64)
            profile[nonzero] = sums[nonzero] / counts[nonzero]
            profile = profile.astype(np.float32)

        for node in psalm.nodes:
            vectors[node] = profile
    return vectors
