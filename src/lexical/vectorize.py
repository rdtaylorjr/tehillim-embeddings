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


def binary_presence_vectors(
    columns: list[PsalmColumns], vocabulary: tuple[str, ...]
) -> dict[int, np.ndarray]:
    """One {0,1} vector, 1 where that half-verse contains the vocabulary entry."""
    counts = term_frequency_vectors(columns, vocabulary)
    return {node: (vector > 0).astype(np.float32) for node, vector in counts.items()}


def log_count_vectors(
    columns: list[PsalmColumns], vocabulary: tuple[str, ...]
) -> dict[int, np.ndarray]:
    """log(1 + term frequency) per half-verse node: repetition matters, damped."""
    counts = term_frequency_vectors(columns, vocabulary)
    return {node: np.log1p(vector).astype(np.float32) for node, vector in counts.items()}


def icf_vector(vocabulary: tuple[str, ...], icf_weights: dict[str, float]) -> np.ndarray:
    """The ICF weight for each vocabulary entry, in vocabulary order."""
    return np.array([icf_weights[value] for value in vocabulary], dtype=np.float32)


def icf_weighted_vectors(
    columns: list[PsalmColumns], vocabulary: tuple[str, ...], icf_weights: dict[str, float]
) -> dict[int, np.ndarray]:
    """Binary presence x ICF(value): a shared rare value scores higher than a common one."""
    weights = icf_vector(vocabulary, icf_weights)
    binary = binary_presence_vectors(columns, vocabulary)
    return {node: vector * weights for node, vector in binary.items()}


def tf_icf_vectors(
    columns: list[PsalmColumns], vocabulary: tuple[str, ...], icf_weights: dict[str, float]
) -> dict[int, np.ndarray]:
    """log(1 + tf) x ICF(value): repetition and rarity combined."""
    weights = icf_vector(vocabulary, icf_weights)
    log_counts = log_count_vectors(columns, vocabulary)
    return {node: vector * weights for node, vector in log_counts.items()}
