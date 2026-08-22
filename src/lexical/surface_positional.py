"""Colon-level ICF-weighted positional pyramid over surface forms, one text tier at a time."""

from __future__ import annotations

import numpy as np

from lexical.positional import bin_index, colon_positions
from lexical.surface_corpus import SurfacePsalm
from lexical.surface_vectorize import surface_icf_vector
from lexical.surface_vocabulary import SurfaceTier, half_verses_for_tier


def surface_positional_icf_vectors(
    psalms: list[SurfacePsalm],
    vocabulary: tuple[str, ...],
    tier: SurfaceTier,
    icf_weights: dict[str, float],
    k: int,
    order_by_psalm: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Per-colon [0;...;own ICF content;...;0]: nonzero only in the colon's own position bin."""
    weights = surface_icf_vector(vocabulary, icf_weights)
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)

    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        half_verses = half_verses_for_tier(psalm, tier)
        n = len(half_verses)
        order = order_by_psalm[psalm.number] if order_by_psalm is not None else np.arange(n)
        bins = bin_index(colon_positions(n), k)

        for position, colon_index in enumerate(order):
            indices = np.fromiter(
                (index_of[v] for v in set(half_verses[colon_index]) if v in index_of),
                dtype=np.int64,
            )
            block = np.zeros((k, dim), dtype=np.float32)
            block[bins[position], indices] = weights[indices]
            vectors[psalm.half_verse_nodes[colon_index]] = block.flatten()
    return vectors
