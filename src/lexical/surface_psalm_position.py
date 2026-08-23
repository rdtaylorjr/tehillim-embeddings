"""Psalm-level ICF-weighted positional pyramid over surface forms, broadcast to every colon."""

from __future__ import annotations

import numpy as np

from lexical.positional import bin_index, colon_positions
from lexical.surface_corpus import SurfacePsalm
from lexical.surface_vectorize import surface_icf_vector
from lexical.surface_vocabulary import SurfaceTier, cola_for_tier


def surface_psalm_positional_icf_vectors(
    psalms: list[SurfacePsalm],
    vocabulary: tuple[str, ...],
    tier: SurfaceTier,
    icf_weights: dict[str, float],
    k: int,
    order_by_psalm: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Psalm-level [B_1;...;B_k] ICF-weighted positional pyramid, broadcast to every colon node."""
    weights = surface_icf_vector(vocabulary, icf_weights)
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)

    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        cola = cola_for_tier(psalm, tier)
        n = len(cola)
        order = order_by_psalm[psalm.number] if order_by_psalm is not None else np.arange(n)
        bins = bin_index(colon_positions(n), k)

        flat_index_parts = []
        flat_weight_parts = []
        for position, colon_index in enumerate(order):
            indices = np.fromiter(
                (index_of[v] for v in set(cola[colon_index]) if v in index_of),
                dtype=np.int64,
            )
            flat_index_parts.append(bins[position] * dim + indices)
            flat_weight_parts.append(weights[indices])

        flat_indices = np.concatenate(flat_index_parts)
        flat_values = np.concatenate(flat_weight_parts)
        psalm_vector = np.bincount(flat_indices, weights=flat_values, minlength=k * dim).astype(
            np.float32
        )
        for node in psalm.colon_nodes:
            vectors[node] = psalm_vector
    return vectors
