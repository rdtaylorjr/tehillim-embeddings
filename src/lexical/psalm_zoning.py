"""Psalm-level frozen ICF inventory concatenated with lexeme-specific mean centered position."""

from __future__ import annotations

import numpy as np

from core.columns import PsalmColumns
from core.position import half_verse_positions
from core.vocabulary import index_map
from lexical.vectorize import icf_vector


def psalm_position_mean_vectors(
    columns: list[PsalmColumns],
    vocabulary: tuple[str, ...],
    icf_weights: dict[str, float],
    order_by_psalm: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Psalm-level: b = ICF if present anywhere, m = ICF x (2 * mean half-verse position - 1)."""
    weights = icf_vector(vocabulary, icf_weights)
    index_of = index_map(vocabulary)
    dim = len(vocabulary)

    vectors: dict[int, np.ndarray] = {}
    for psalm in columns:
        half_verses = psalm.half_verses
        n = len(half_verses)
        order = order_by_psalm[psalm.number] if order_by_psalm is not None else np.arange(n)
        ordered = [half_verses[i] for i in order]
        t = half_verse_positions(n)

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
        for node in psalm.nodes:
            vectors[node] = psalm_vector
    return vectors
