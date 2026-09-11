"""Psalm-scale lexical baselines: counted over the whole psalm, then broadcast to its nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from core.vocabulary import index_map
from lexical.vectorize import damped, icf_vector, presence_of, scaled_by

if TYPE_CHECKING:
    from core.columns import PsalmColumns


def psalm_term_frequency_vectors(
    columns: list[PsalmColumns], vocabulary: tuple[str, ...]
) -> dict[int, np.ndarray]:
    """Raw occurrence counts over the whole psalm, broadcast to every half-verse node."""
    index_of = index_map(vocabulary)
    dim = len(vocabulary)
    vectors: dict[int, np.ndarray] = {}
    for psalm in columns:
        indices = np.fromiter(
            (
                index_of[value]
                for half_verse in psalm.half_verses
                for value in half_verse
                if value in index_of
            ),
            dtype=np.int64,
        )
        counts = np.bincount(indices, minlength=dim).astype(np.float32)
        for node in psalm.nodes:
            vectors[node] = counts
    return vectors


def psalm_binary_presence_vectors(
    columns: list[PsalmColumns], vocabulary: tuple[str, ...]
) -> dict[int, np.ndarray]:
    """One {0,1} vector, 1 where the psalm contains the vocabulary entry anywhere."""
    return presence_of(psalm_term_frequency_vectors(columns, vocabulary))


def psalm_log_count_vectors(
    columns: list[PsalmColumns], vocabulary: tuple[str, ...]
) -> dict[int, np.ndarray]:
    """log(1 + psalm term frequency): repetition across the psalm matters, damped."""
    return damped(psalm_term_frequency_vectors(columns, vocabulary))


def psalm_icf_weighted_vectors(
    columns: list[PsalmColumns], vocabulary: tuple[str, ...], icf_weights: dict[str, float]
) -> dict[int, np.ndarray]:
    """Psalm-level binary presence x ICF(value)."""
    return scaled_by(
        psalm_binary_presence_vectors(columns, vocabulary), icf_vector(vocabulary, icf_weights)
    )


def psalm_tf_icf_vectors(
    columns: list[PsalmColumns], vocabulary: tuple[str, ...], icf_weights: dict[str, float]
) -> dict[int, np.ndarray]:
    """Psalm-level log(1 + tf) x ICF(value)."""
    return scaled_by(
        psalm_log_count_vectors(columns, vocabulary), icf_vector(vocabulary, icf_weights)
    )
