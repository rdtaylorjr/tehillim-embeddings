"""Converts half-verse value sequences into per-half-verse vectors: binary, count, ICF-weighted."""

from __future__ import annotations

import numpy as np

from core.columns import PsalmColumns
from core.vocabulary import index_map


def term_frequency_vectors(
    columns: list[PsalmColumns], vocabulary: tuple[str, ...]
) -> dict[int, np.ndarray]:
    """One raw occurrence-count vector per half-verse node; every weighting derives from this."""
    index_of = index_map(vocabulary)
    dim = len(vocabulary)
    vectors: dict[int, np.ndarray] = {}
    for psalm in columns:
        for node, half_verse in zip(psalm.nodes, psalm.half_verses, strict=True):
            indices = np.fromiter(
                (index_of[v] for v in half_verse if v in index_of), dtype=np.int64
            )
            vectors[node] = np.bincount(indices, minlength=dim).astype(np.float32)
    return vectors


def presence_of(counts: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """The {0,1} form of any count map, whatever scale the counts were taken at."""
    return {node: (vector > 0).astype(np.float32) for node, vector in counts.items()}


def damped(counts: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """log(1 + count): repetition matters, damped, at whatever scale the counts were taken."""
    return {node: np.log1p(vector).astype(np.float32) for node, vector in counts.items()}


def scaled_by(vectors: dict[int, np.ndarray], weights: np.ndarray) -> dict[int, np.ndarray]:
    """Each vector multiplied entrywise by one shared weight vector."""
    return {node: vector * weights for node, vector in vectors.items()}


def binary_presence_vectors(
    columns: list[PsalmColumns], vocabulary: tuple[str, ...]
) -> dict[int, np.ndarray]:
    """One {0,1} vector, 1 where that half-verse contains the vocabulary entry."""
    return presence_of(term_frequency_vectors(columns, vocabulary))


def log_count_vectors(
    columns: list[PsalmColumns], vocabulary: tuple[str, ...]
) -> dict[int, np.ndarray]:
    """log(1 + term frequency) per half-verse node: repetition matters, damped."""
    return damped(term_frequency_vectors(columns, vocabulary))


def icf_vector(vocabulary: tuple[str, ...], icf_weights: dict[str, float]) -> np.ndarray:
    """The ICF weight for each vocabulary entry, in vocabulary order."""
    return np.array([icf_weights[value] for value in vocabulary], dtype=np.float32)


def icf_weighted_vectors(
    columns: list[PsalmColumns], vocabulary: tuple[str, ...], icf_weights: dict[str, float]
) -> dict[int, np.ndarray]:
    """Binary presence x ICF(value): a shared rare value scores higher than a common one."""
    return scaled_by(
        binary_presence_vectors(columns, vocabulary), icf_vector(vocabulary, icf_weights)
    )


def tf_icf_vectors(
    columns: list[PsalmColumns], vocabulary: tuple[str, ...], icf_weights: dict[str, float]
) -> dict[int, np.ndarray]:
    """log(1 + tf) x ICF(value): repetition and rarity combined."""
    return scaled_by(log_count_vectors(columns, vocabulary), icf_vector(vocabulary, icf_weights))
