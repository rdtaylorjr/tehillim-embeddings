"""Converts colon lex/lex0 sequences into fixed-length binary presence vectors over a vocabulary."""

from __future__ import annotations

import numpy as np

from lexical.corpus import LexicalPsalm
from lexical.vocabulary import VocabularyKey, half_verses_for_key


def binary_presence_vectors(
    psalms: list[LexicalPsalm], vocabulary: tuple[str, ...], key: VocabularyKey
) -> dict[int, np.ndarray]:
    """One {0,1} vector per half-verse node, 1 where that colon contains the vocabulary entry."""
    index_of = {value: i for i, value in enumerate(vocabulary)}
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        half_verses = half_verses_for_key(psalm, key)
        for node, half_verse in zip(psalm.half_verse_nodes, half_verses, strict=True):
            vector = np.zeros(len(vocabulary), dtype=np.float32)
            for value in half_verse:
                index = index_of.get(value)
                if index is not None:
                    vector[index] = 1.0
            vectors[node] = vector
    return vectors
