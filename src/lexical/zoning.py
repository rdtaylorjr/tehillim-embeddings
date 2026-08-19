"""Psalm-level frozen ICF inventory concatenated with lexeme-specific mean centered position."""

from __future__ import annotations

import numpy as np

from lexical.corpus import LexicalPsalm
from lexical.positional import colon_positions
from lexical.vectorize import icf_vector
from lexical.vocabulary import VocabularyKey, half_verses_for_key


def positional_centroid_vectors(
    psalms: list[LexicalPsalm],
    vocabulary: tuple[str, ...],
    key: VocabularyKey,
    icf_weights: dict[str, float],
    order_by_psalm: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Psalm-level [b; m]: b = ICF if present anywhere, m = ICF x (2 * mean colon position - 1)."""
    weights = icf_vector(vocabulary, icf_weights)
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)

    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        half_verses = half_verses_for_key(psalm, key)
        n = len(half_verses)
        order = order_by_psalm[psalm.number] if order_by_psalm is not None else np.arange(n)
        ordered = [half_verses[i] for i in order]
        t = colon_positions(n)

        position_sums = np.zeros(dim, dtype=np.float64)
        position_counts = np.zeros(dim, dtype=np.float64)
        for position, half_verse in enumerate(ordered):
            for value in set(half_verse):
                index = index_of.get(value)
                if index is not None:
                    position_sums[index] += t[position]
                    position_counts[index] += 1.0

        present = position_counts > 0
        b = weights * present
        mean_position = np.divide(position_sums, position_counts, out=np.zeros(dim), where=present)
        m = weights * present * (2 * mean_position - 1)

        psalm_vector = np.concatenate([b, m]).astype(np.float32)
        for node in psalm.half_verse_nodes:
            vectors[node] = psalm_vector
    return vectors
