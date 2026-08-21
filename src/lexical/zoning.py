"""Per-colon frozen ICF inventory concatenated with that colon's own centered position."""

from __future__ import annotations

import numpy as np

from lexical.corpus import LexicalPsalm
from lexical.positional import colon_positions
from lexical.vectorize import icf_vector
from lexical.vocabulary import VocabularyKey, half_verses_for_key


def position_mean_vectors(
    psalms: list[LexicalPsalm],
    vocabulary: tuple[str, ...],
    key: VocabularyKey,
    icf_weights: dict[str, float],
    order_by_psalm: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Per-colon [b; m]: b = ICF if present in this colon, m = ICF x (2 * this colon's t - 1)."""
    weights = icf_vector(vocabulary, icf_weights)
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)

    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        half_verses = half_verses_for_key(psalm, key)
        n = len(half_verses)
        order = order_by_psalm[psalm.number] if order_by_psalm is not None else np.arange(n)
        t = colon_positions(n)

        for position, colon_index in enumerate(order):
            present = np.zeros(dim, dtype=bool)
            for value in set(half_verses[colon_index]):
                index = index_of.get(value)
                if index is not None:
                    present[index] = True

            b = weights * present
            m = weights * present * (2 * t[position] - 1)
            colon_vector = np.concatenate([b, m]).astype(np.float32)
            vectors[psalm.half_verse_nodes[colon_index]] = colon_vector
    return vectors
