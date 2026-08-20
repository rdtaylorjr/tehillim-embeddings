"""Psalm-level ICF-weighted positional pyramid, broadcast to every colon node in the psalm."""

from __future__ import annotations

import numpy as np

from lexical.corpus import LexicalPsalm
from lexical.positional import bin_index, colon_positions
from lexical.vectorize import icf_vector
from lexical.vocabulary import VocabularyKey, half_verses_for_key


def psalm_positional_icf_vectors(
    psalms: list[LexicalPsalm],
    vocabulary: tuple[str, ...],
    key: VocabularyKey,
    icf_weights: dict[str, float],
    k: int,
    order_by_psalm: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Psalm-level [B_1;...;B_k] ICF-weighted positional pyramid, broadcast to every colon node."""
    weights = icf_vector(vocabulary, icf_weights)
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)

    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        half_verses = half_verses_for_key(psalm, key)
        n = len(half_verses)
        order = order_by_psalm[psalm.number] if order_by_psalm is not None else np.arange(n)
        bins = bin_index(colon_positions(n), k)

        blocks = np.zeros((k, dim), dtype=np.float32)
        for position, colon_index in enumerate(order):
            for value in set(half_verses[colon_index]):
                index = index_of.get(value)
                if index is not None:
                    blocks[bins[position], index] += weights[index]

        psalm_vector = blocks.flatten()
        for node in psalm.half_verse_nodes:
            vectors[node] = psalm_vector
    return vectors
