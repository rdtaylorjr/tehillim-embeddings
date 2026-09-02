"""Per-half-verse frozen ICF inventory concatenated with its own centered position."""

from __future__ import annotations

import numpy as np

from core.columns import PsalmColumns
from core.position import half_verse_positions
from core.vocabulary import index_map
from lexical.vectorize import icf_vector


def position_mean_vectors(
    columns: list[PsalmColumns],
    vocabulary: tuple[str, ...],
    icf_weights: dict[str, float],
    order_by_psalm: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Per-half-verse [b; m]: b = ICF if present here, m = ICF x (2t - 1)."""
    weights = icf_vector(vocabulary, icf_weights)
    index_of = index_map(vocabulary)
    dim = len(vocabulary)

    vectors: dict[int, np.ndarray] = {}
    for psalm in columns:
        half_verses = psalm.half_verses
        n = len(half_verses)
        order = order_by_psalm[psalm.number] if order_by_psalm is not None else np.arange(n)
        t = half_verse_positions(n)

        for position, half_verse_index in enumerate(order):
            indices = np.fromiter(
                (index_of[v] for v in set(half_verses[half_verse_index]) if v in index_of),
                dtype=np.int64,
            )
            present = np.zeros(dim, dtype=bool)
            present[indices] = True

            b = weights * present
            m = weights * present * (2 * t[position] - 1)
            half_verse_vector = np.concatenate([b, m]).astype(np.float32)
            vectors[psalm.nodes[half_verse_index]] = half_verse_vector
    return vectors
