"""Psalm-level frozen ICF inventory over surface forms, concatenated with mean centered position."""

from __future__ import annotations

import numpy as np

from lexical.positional import colon_positions
from lexical.surface_corpus import SurfacePsalm
from lexical.surface_vectorize import surface_icf_vector
from lexical.surface_vocabulary import SurfaceTier, half_verses_for_tier


def surface_psalm_position_mean_vectors(
    psalms: list[SurfacePsalm],
    vocabulary: tuple[str, ...],
    tier: SurfaceTier,
    icf_weights: dict[str, float],
    order_by_psalm: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Psalm-level [b; m]: b = ICF if present anywhere, m = ICF x (2 * mean colon position - 1)."""
    weights = surface_icf_vector(vocabulary, icf_weights)
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)

    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        half_verses = half_verses_for_tier(psalm, tier)
        n = len(half_verses)
        order = order_by_psalm[psalm.number] if order_by_psalm is not None else np.arange(n)
        ordered = [half_verses[i] for i in order]
        t = colon_positions(n)

        flat_index_parts = []
        flat_t_parts = []
        for position, half_verse in enumerate(ordered):
            indices = np.fromiter(
                (index_of[v] for v in set(half_verse) if v in index_of), dtype=np.int64
            )
            flat_index_parts.append(indices)
            flat_t_parts.append(np.full(len(indices), t[position]))
        flat_indices = np.concatenate(flat_index_parts)
        flat_t = np.concatenate(flat_t_parts)

        position_sums = np.bincount(flat_indices, weights=flat_t, minlength=dim)
        position_counts = np.bincount(flat_indices, minlength=dim).astype(np.float64)

        present = position_counts > 0
        b = weights * present
        mean_position = np.divide(position_sums, position_counts, out=np.zeros(dim), where=present)
        m = weights * present * (2 * mean_position - 1)

        psalm_vector = np.concatenate([b, m]).astype(np.float32)
        for node in psalm.half_verse_nodes:
            vectors[node] = psalm_vector
    return vectors
