"""Per-feature morphology histograms: normalized proportions, NA as part of the distribution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

from core.ngram import unigram_histogram
from core.vocabulary import index_map
from morphology.corpus import MorphologicalPsalm
from morphology.pos_ngram import pos_unigram_histogram, sp_unigram_psalm_vectors
from morphology.vocabulary import (
    GN_VOCABULARY,
    NU_VOCABULARY,
    PRS_GN_VOCABULARY,
    PRS_NU_VOCABULARY,
    PRS_PS_VOCABULARY,
    PS_VOCABULARY,
    ST_VOCABULARY,
    VS_VOCABULARY,
    VT_VOCABULARY,
)

FeatureKey = Literal["gn", "nu", "ps", "st", "vs", "vt", "prs_gn", "prs_nu", "prs_ps"]

_VOCABULARY_BY_FEATURE: dict[FeatureKey, tuple[str, ...]] = {
    "gn": GN_VOCABULARY,
    "nu": NU_VOCABULARY,
    "ps": PS_VOCABULARY,
    "st": ST_VOCABULARY,
    "vs": VS_VOCABULARY,
    "vt": VT_VOCABULARY,
    "prs_gn": PRS_GN_VOCABULARY,
    "prs_nu": PRS_NU_VOCABULARY,
    "prs_ps": PRS_PS_VOCABULARY,
}

_FULL_FEATURE_ORDER: tuple[FeatureKey, ...] = (
    "gn",
    "nu",
    "ps",
    "st",
    "vs",
    "vt",
    "prs_gn",
    "prs_nu",
    "prs_ps",
)


#: Written out rather than resolved by name so a renamed field fails type checking.
_HALF_VERSES_BY_FEATURE: dict[
    FeatureKey, Callable[[MorphologicalPsalm], tuple[tuple[str, ...], ...]]
] = {
    "gn": lambda psalm: psalm.half_verse_gn,
    "nu": lambda psalm: psalm.half_verse_nu,
    "ps": lambda psalm: psalm.half_verse_ps,
    "st": lambda psalm: psalm.half_verse_st,
    "vs": lambda psalm: psalm.half_verse_vs,
    "vt": lambda psalm: psalm.half_verse_vt,
    "prs_gn": lambda psalm: psalm.half_verse_prs_gn,
    "prs_nu": lambda psalm: psalm.half_verse_prs_nu,
    "prs_ps": lambda psalm: psalm.half_verse_prs_ps,
}


def half_verses_for_feature(
    psalm: MorphologicalPsalm, feature: FeatureKey
) -> tuple[tuple[str, ...], ...]:
    """Selects one psalm's per-half-verse value sequences for `feature`."""
    return _HALF_VERSES_BY_FEATURE[feature](psalm)


def atomic_histogram(half_verse_values: tuple[str, ...], vocabulary: tuple[str, ...]) -> np.ndarray:
    """Normalized value proportions over one node: count(v) / m, NA included as its own bin."""
    index_of = index_map(vocabulary)
    return unigram_histogram(half_verse_values, index_of, len(vocabulary))


def atomic_vectors(psalms: list[MorphologicalPsalm], feature: FeatureKey) -> dict[int, np.ndarray]:
    """One atomic histogram per half-verse node for `feature`."""
    vocabulary = _VOCABULARY_BY_FEATURE[feature]
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        half_verses = half_verses_for_feature(psalm, feature)
        for node, half_verse_values in zip(psalm.half_verse_nodes, half_verses, strict=True):
            vectors[node] = atomic_histogram(half_verse_values, vocabulary)
    return vectors


def atomic_psalm_vectors(
    psalms: list[MorphologicalPsalm], feature: FeatureKey
) -> dict[int, np.ndarray]:
    """Psalm-broadcast atomic histogram: word-count-weighted pooling across every half-verse."""
    vocabulary = _VOCABULARY_BY_FEATURE[feature]
    index_of = index_map(vocabulary)
    dim = len(vocabulary)
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        half_verses = half_verses_for_feature(psalm, feature)
        flattened = tuple(value for half_verse_values in half_verses for value in half_verse_values)
        psalm_vector = unigram_histogram(flattened, index_of, dim)
        for node in psalm.half_verse_nodes:
            vectors[node] = psalm_vector
    return vectors


def sp_plus_feature_vectors(
    psalms: list[MorphologicalPsalm], feature: FeatureKey
) -> dict[int, np.ndarray]:
    """`[sp_unigram; atomic(feature)]` per half-verse node."""
    feature_vectors = atomic_vectors(psalms, feature)
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, half_verse_sp in zip(psalm.half_verse_nodes, psalm.half_verse_sp, strict=True):
            vectors[node] = np.concatenate(
                [pos_unigram_histogram(half_verse_sp), feature_vectors[node]]
            )
    return vectors


def sp_plus_feature_psalm_vectors(
    psalms: list[MorphologicalPsalm], feature: FeatureKey
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `[sp_unigram; atomic(feature)]`."""
    sp_vectors = sp_unigram_psalm_vectors(psalms)
    feature_vectors = atomic_psalm_vectors(psalms, feature)
    return {node: np.concatenate([sp_vectors[node], feature_vectors[node]]) for node in sp_vectors}


def full_morphology_vectors(psalms: list[MorphologicalPsalm]) -> dict[int, np.ndarray]:
    """[sp; gn; nu; ps; st; vs; vt; prs_gn; prs_nu; prs_ps] per node, dim 77 (H4.4 baseline)."""
    per_feature = {feature: atomic_vectors(psalms, feature) for feature in _FULL_FEATURE_ORDER}
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, half_verse_sp in zip(psalm.half_verse_nodes, psalm.half_verse_sp, strict=True):
            blocks = [pos_unigram_histogram(half_verse_sp)]
            blocks.extend(per_feature[feature][node] for feature in _FULL_FEATURE_ORDER)
            vectors[node] = np.concatenate(blocks)
    return vectors


def full_morphology_psalm_vectors(psalms: list[MorphologicalPsalm]) -> dict[int, np.ndarray]:
    """Psalm-broadcast `full_morphology_vectors`."""
    sp_vectors = sp_unigram_psalm_vectors(psalms)
    per_feature = {
        feature: atomic_psalm_vectors(psalms, feature) for feature in _FULL_FEATURE_ORDER
    }
    vectors: dict[int, np.ndarray] = {}
    for node in sp_vectors:
        blocks = [sp_vectors[node]]
        blocks.extend(per_feature[feature][node] for feature in _FULL_FEATURE_ORDER)
        vectors[node] = np.concatenate(blocks)
    return vectors
