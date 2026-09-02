"""Half-verse-level ICF-weighted positional pyramid: each nonzero only in its own region."""

from __future__ import annotations

import numpy as np

from core.columns import PsalmColumns
from core.position import bin_index, half_verse_positions
from core.vocabulary import index_map
from lexical.vectorize import icf_vector


def positional_icf_vectors(
    columns: list[PsalmColumns],
    vocabulary: tuple[str, ...],
    icf_weights: dict[str, float],
    k: int,
    order_by_psalm: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Per-half-verse [0;...;own ICF content;...;0]: nonzero only in its own position bin."""
    weights = icf_vector(vocabulary, icf_weights)
    index_of = index_map(vocabulary)
    dim = len(vocabulary)

    vectors: dict[int, np.ndarray] = {}
    for psalm in columns:
        half_verses = psalm.half_verses
        n = len(half_verses)
        order = order_by_psalm[psalm.number] if order_by_psalm is not None else np.arange(n)
        bins = bin_index(half_verse_positions(n), k)

        for position, half_verse_index in enumerate(order):
            indices = np.fromiter(
                (index_of[v] for v in set(half_verses[half_verse_index]) if v in index_of),
                dtype=np.int64,
            )
            block = np.zeros((k, dim), dtype=np.float32)
            block[bins[position], indices] = weights[indices]
            vectors[psalm.nodes[half_verse_index]] = block.flatten()
    return vectors
