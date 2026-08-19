"""Converts colon lex/lex0 sequences into per-colon vectors: binary, count, ICF-weighted."""

from __future__ import annotations

import numpy as np

from lexical.corpus import LexicalPsalm
from lexical.vocabulary import VocabularyKey, half_verses_for_key


def term_frequency_vectors(
    psalms: list[LexicalPsalm], vocabulary: tuple[str, ...], key: VocabularyKey
) -> dict[int, np.ndarray]:
    """One raw occurrence-count vector per half-verse node; every weighting derives from this."""
    index_of = {value: i for i, value in enumerate(vocabulary)}
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        half_verses = half_verses_for_key(psalm, key)
        for node, half_verse in zip(psalm.half_verse_nodes, half_verses, strict=True):
            vector = np.zeros(len(vocabulary), dtype=np.float32)
            for value in half_verse:
                index = index_of.get(value)
                if index is not None:
                    vector[index] += 1.0
            vectors[node] = vector
    return vectors


def binary_presence_vectors(
    psalms: list[LexicalPsalm], vocabulary: tuple[str, ...], key: VocabularyKey
) -> dict[int, np.ndarray]:
    """One {0,1} vector per half-verse node, 1 where that colon contains the vocabulary entry."""
    counts = term_frequency_vectors(psalms, vocabulary, key)
    return {node: (vector > 0).astype(np.float32) for node, vector in counts.items()}


def log_count_vectors(
    psalms: list[LexicalPsalm], vocabulary: tuple[str, ...], key: VocabularyKey
) -> dict[int, np.ndarray]:
    """log(1 + term frequency) per half-verse node: repetition matters, damped."""
    counts = term_frequency_vectors(psalms, vocabulary, key)
    return {node: np.log1p(vector).astype(np.float32) for node, vector in counts.items()}


def _icf_vector(vocabulary: tuple[str, ...], icf_weights: dict[str, float]) -> np.ndarray:
    """The ICF weight for each vocabulary entry, in vocabulary order."""
    return np.array([icf_weights[value] for value in vocabulary], dtype=np.float32)


def icf_weighted_vectors(
    psalms: list[LexicalPsalm],
    vocabulary: tuple[str, ...],
    key: VocabularyKey,
    icf_weights: dict[str, float],
) -> dict[int, np.ndarray]:
    """Binary presence x ICF(lexeme): a shared rare lexeme scores higher than a common one."""
    weights = _icf_vector(vocabulary, icf_weights)
    binary = binary_presence_vectors(psalms, vocabulary, key)
    return {node: vector * weights for node, vector in binary.items()}


def tf_icf_vectors(
    psalms: list[LexicalPsalm],
    vocabulary: tuple[str, ...],
    key: VocabularyKey,
    icf_weights: dict[str, float],
) -> dict[int, np.ndarray]:
    """log(1 + tf) x ICF(lexeme): repetition and rarity combined."""
    weights = _icf_vector(vocabulary, icf_weights)
    log_counts = log_count_vectors(psalms, vocabulary, key)
    return {node: vector * weights for node, vector in log_counts.items()}
