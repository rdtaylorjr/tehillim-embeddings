"""Psalm-scale grammatical deployment: uniform-weight inventory concatenated with mean position."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from lexical.positional import colon_positions
from morphology.corpus import MorphologicalPsalm
from morphology.suffix import SUFFIX_VOCABULARY, psalm_suffix_signatures

ColonValuesByPsalm = Callable[[MorphologicalPsalm], tuple[tuple[str, ...], ...]]


def psalm_deploy_vectors(
    psalms: list[MorphologicalPsalm],
    vocabulary: tuple[str, ...],
    colon_values_by_psalm: ColonValuesByPsalm,
    order_by_psalm: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Psalm-level [b; m]: b = 1.0 if present anywhere, m = present * (2 * mean position - 1)."""
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)

    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        cola = colon_values_by_psalm(psalm)
        n = len(cola)
        order = order_by_psalm[psalm.number] if order_by_psalm is not None else np.arange(n)
        ordered = [cola[i] for i in order]
        t = colon_positions(n)

        flat_index_parts = []
        flat_t_parts = []
        for position, colon_values in enumerate(ordered):
            indices = np.fromiter(
                (index_of[v] for v in set(colon_values) if v in index_of), dtype=np.int64
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
        for node in psalm.colon_nodes:
            vectors[node] = psalm_vector
    return vectors


def suffix_deploy_vectors(
    psalms: list[MorphologicalPsalm], order_by_psalm: dict[int, np.ndarray] | None = None
) -> dict[int, np.ndarray]:
    """`psalm_deploy_vectors` over the pronominal-suffix vocabulary, the 4B-4D-strongest signal."""
    return psalm_deploy_vectors(psalms, SUFFIX_VOCABULARY, psalm_suffix_signatures, order_by_psalm)
