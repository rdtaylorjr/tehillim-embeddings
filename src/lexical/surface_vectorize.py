"""Converts colon surface-form sequences into per-colon vectors: binary, count, ICF-weighted."""

from __future__ import annotations

import numpy as np

from lexical.surface_corpus import SurfacePsalm
from lexical.surface_vocabulary import SurfaceTier, half_verses_for_tier


def surface_term_frequency_vectors(
    psalms: list[SurfacePsalm], vocabulary: tuple[str, ...], tier: SurfaceTier
) -> dict[int, np.ndarray]:
    """One raw occurrence-count vector per half-verse node; every weighting derives from this."""
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        half_verses = half_verses_for_tier(psalm, tier)
        for node, half_verse in zip(psalm.half_verse_nodes, half_verses, strict=True):
            indices = np.fromiter(
                (index_of[v] for v in half_verse if v in index_of), dtype=np.int64
            )
            vectors[node] = np.bincount(indices, minlength=dim).astype(np.float32)
    return vectors


def surface_binary_presence_vectors(
    psalms: list[SurfacePsalm], vocabulary: tuple[str, ...], tier: SurfaceTier
) -> dict[int, np.ndarray]:
    """One {0,1} vector per half-verse node, 1 where that colon contains the vocabulary entry."""
    counts = surface_term_frequency_vectors(psalms, vocabulary, tier)
    return {node: (vector > 0).astype(np.float32) for node, vector in counts.items()}


def surface_log_count_vectors(
    psalms: list[SurfacePsalm], vocabulary: tuple[str, ...], tier: SurfaceTier
) -> dict[int, np.ndarray]:
    """log(1 + term frequency) per half-verse node: repetition matters, damped."""
    counts = surface_term_frequency_vectors(psalms, vocabulary, tier)
    return {node: np.log1p(vector).astype(np.float32) for node, vector in counts.items()}


def surface_icf_vector(vocabulary: tuple[str, ...], icf_weights: dict[str, float]) -> np.ndarray:
    """The ICF weight for each vocabulary entry, in vocabulary order."""
    return np.array([icf_weights[value] for value in vocabulary], dtype=np.float32)


def surface_icf_weighted_vectors(
    psalms: list[SurfacePsalm],
    vocabulary: tuple[str, ...],
    tier: SurfaceTier,
    icf_weights: dict[str, float],
) -> dict[int, np.ndarray]:
    """Binary presence x ICF(form): a shared rare surface form scores higher than a common one."""
    weights = surface_icf_vector(vocabulary, icf_weights)
    binary = surface_binary_presence_vectors(psalms, vocabulary, tier)
    return {node: vector * weights for node, vector in binary.items()}


def surface_tf_icf_vectors(
    psalms: list[SurfacePsalm],
    vocabulary: tuple[str, ...],
    tier: SurfaceTier,
    icf_weights: dict[str, float],
) -> dict[int, np.ndarray]:
    """log(1 + tf) x ICF(form): repetition and rarity combined."""
    weights = surface_icf_vector(vocabulary, icf_weights)
    log_counts = surface_log_count_vectors(psalms, vocabulary, tier)
    return {node: vector * weights for node, vector in log_counts.items()}
