"""Colon-level ICF-weighted positional pyramid: each colon nonzero only in its own region."""

from __future__ import annotations

import numpy as np

from lexical.corpus import LexicalPsalm
from lexical.vectorize import icf_vector
from lexical.vocabulary import VocabularyKey, half_verses_for_key


def colon_positions(n: int) -> np.ndarray:
    """Continuity-corrected normalized colon position t_i = (i - 0.5) / n, for i = 1..n."""
    return (np.arange(1, n + 1) - 0.5) / n


def bin_index(t: np.ndarray, k: int) -> np.ndarray:
    """Which of k equal-width [0, 1) regions each normalized position falls into."""
    return np.asarray(np.minimum((t * k).astype(int), k - 1))


def positional_icf_vectors(
    psalms: list[LexicalPsalm],
    vocabulary: tuple[str, ...],
    key: VocabularyKey,
    icf_weights: dict[str, float],
    k: int,
    order_by_psalm: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Per-colon [0;...;own ICF content;...;0]: nonzero only in the colon's own position bin."""
    weights = icf_vector(vocabulary, icf_weights)
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)

    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        half_verses = half_verses_for_key(psalm, key)
        n = len(half_verses)
        order = order_by_psalm[psalm.number] if order_by_psalm is not None else np.arange(n)
        bins = bin_index(colon_positions(n), k)

        for position, colon_index in enumerate(order):
            block = np.zeros((k, dim), dtype=np.float32)
            for value in set(half_verses[colon_index]):
                index = index_of.get(value)
                if index is not None:
                    block[bins[position], index] += weights[index]
            vectors[psalm.half_verse_nodes[colon_index]] = block.flatten()
    return vectors
