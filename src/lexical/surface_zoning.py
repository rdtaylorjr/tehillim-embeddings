"""Per-colon frozen ICF inventory over surface forms, concatenated with centered position."""

from __future__ import annotations

import numpy as np

from lexical.positional import colon_positions
from lexical.surface_corpus import SurfacePsalm
from lexical.surface_vectorize import surface_icf_vector
from lexical.surface_vocabulary import SurfaceTier, cola_for_tier


def surface_position_mean_vectors(
    psalms: list[SurfacePsalm],
    vocabulary: tuple[str, ...],
    tier: SurfaceTier,
    icf_weights: dict[str, float],
    order_by_psalm: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Per-colon [b; m]: b = ICF if present in this colon, m = ICF x (2 * this colon's t - 1)."""
    weights = surface_icf_vector(vocabulary, icf_weights)
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)

    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        cola = cola_for_tier(psalm, tier)
        n = len(cola)
        order = order_by_psalm[psalm.number] if order_by_psalm is not None else np.arange(n)
        t = colon_positions(n)

        for position, colon_index in enumerate(order):
            indices = np.fromiter(
                (index_of[v] for v in set(cola[colon_index]) if v in index_of),
                dtype=np.int64,
            )
            present = np.zeros(dim, dtype=bool)
            present[indices] = True

            b = weights * present
            m = weights * present * (2 * t[position] - 1)
            colon_vector = np.concatenate([b, m]).astype(np.float32)
            vectors[psalm.colon_nodes[colon_index]] = colon_vector
    return vectors
