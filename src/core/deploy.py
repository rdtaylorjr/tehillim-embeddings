"""Psalm-scale deployment vectors: a uniform-weight inventory beside each value's mean position."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from core.position import half_verse_positions
from core.shuffle import NumberedPsalm
from core.vocabulary import index_map

if TYPE_CHECKING:
    from collections.abc import Callable


def psalm_deploy_vectors[PsalmT: NumberedPsalm](
    psalms: list[PsalmT],
    vocabulary: tuple[str, ...],
    half_verse_values_by_psalm: Callable[[PsalmT], tuple[tuple[str, ...], ...]],
    order_by_psalm: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Psalm-level [b; m]: b = 1.0 if present anywhere, m = present * (2 * mean position - 1)."""
    index_of = index_map(vocabulary)
    dim = len(vocabulary)

    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        half_verses = half_verse_values_by_psalm(psalm)
        n = len(half_verses)
        order = order_by_psalm[psalm.number] if order_by_psalm is not None else np.arange(n)
        ordered = [half_verses[i] for i in order]
        t = half_verse_positions(n)

        flat_index_parts = []
        flat_t_parts = []
        for position, half_verse_values in enumerate(ordered):
            indices = np.fromiter(
                (index_of[v] for v in set(half_verse_values) if v in index_of), dtype=np.int64
            )
            flat_index_parts.append(indices)
            flat_t_parts.append(np.full(len(indices), t[position]))
        flat_indices = np.concatenate(flat_index_parts)
        flat_t = np.concatenate(flat_t_parts)

        position_sums = np.bincount(flat_indices, weights=flat_t, minlength=dim)
        position_counts = np.bincount(flat_indices, minlength=dim).astype(np.float64)

        present = position_counts > 0
        b = present.astype(np.float64)
        mean_position = np.divide(position_sums, position_counts, out=np.zeros(dim), where=present)
        m = present * (2 * mean_position - 1)

        psalm_vector = np.concatenate([b, m]).astype(np.float32)
        for node in psalm.half_verse_nodes:
            vectors[node] = psalm_vector
    return vectors
