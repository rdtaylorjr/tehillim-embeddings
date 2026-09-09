"""Psalm-level ICF-weighted positional pyramid, broadcast to every half-verse of the psalm."""

from __future__ import annotations

import numpy as np

from core.columns import PsalmColumns
from core.position import bin_index, half_verse_positions
from core.vocabulary import index_map
from lexical.vectorize import icf_vector


def psalm_positional_icf_vectors(
    columns: list[PsalmColumns],
    vocabulary: tuple[str, ...],
    icf_weights: dict[str, float],
    k: int,
    order_by_psalm: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Psalm-level ICF-weighted positional pyramid, broadcast to every half-verse node."""
    weights = icf_vector(vocabulary, icf_weights)
    index_of = index_map(vocabulary)
    dim = len(vocabulary)

    vectors: dict[int, np.ndarray] = {}
    for psalm in columns:
        half_verses = psalm.half_verses
        n = len(half_verses)
        order = order_by_psalm[psalm.number] if order_by_psalm is not None else np.arange(n)
        bins = bin_index(half_verse_positions(n), k)

        flat_index_parts = []
        flat_weight_parts = []
        for position, half_verse_index in enumerate(order):
            indices = np.fromiter(
                (index_of[v] for v in set(half_verses[half_verse_index]) if v in index_of),
                dtype=np.int64,
            )
            flat_index_parts.append(bins[position] * dim + indices)
            flat_weight_parts.append(weights[indices])

        flat_indices = np.concatenate(flat_index_parts)
        flat_values = np.concatenate(flat_weight_parts)
        psalm_vector = np.bincount(flat_indices, weights=flat_values, minlength=k * dim).astype(
            np.float32
        )
        for node in psalm.nodes:
            vectors[node] = psalm_vector
    return vectors
