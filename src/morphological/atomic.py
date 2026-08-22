"""Per-feature morphology histograms: normalized proportions, NA as part of the distribution."""

from __future__ import annotations

from typing import Literal

import numpy as np

from morphological.corpus import MorphologicalPsalm
from morphological.ngram import unigram_histogram
from morphological.pos_ngram import pos_unigram_histogram, sp_unigram_psalm_vectors
from morphological.vocabulary import (
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


def _half_verses_for_feature(
    psalm: MorphologicalPsalm, feature: FeatureKey
) -> tuple[tuple[str, ...], ...]:
    if feature == "gn":
        return psalm.half_verse_gn
    if feature == "nu":
        return psalm.half_verse_nu
    if feature == "ps":
        return psalm.half_verse_ps
    if feature == "st":
        return psalm.half_verse_st
    if feature == "vs":
        return psalm.half_verse_vs
    if feature == "vt":
        return psalm.half_verse_vt
    if feature == "prs_gn":
        return psalm.half_verse_prs_gn
    if feature == "prs_nu":
        return psalm.half_verse_prs_nu
    return psalm.half_verse_prs_ps


def atomic_histogram(colon_values: tuple[str, ...], vocabulary: tuple[str, ...]) -> np.ndarray:
    """Normalized value proportions over one colon: count(v) / m, NA included as its own bin."""
    index_of = {value: i for i, value in enumerate(vocabulary)}
    return unigram_histogram(colon_values, index_of, len(vocabulary))


def atomic_vectors(psalms: list[MorphologicalPsalm], feature: FeatureKey) -> dict[int, np.ndarray]:
    """One atomic histogram per colon node for `feature`."""
    vocabulary = _VOCABULARY_BY_FEATURE[feature]
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        half_verses = _half_verses_for_feature(psalm, feature)
        for node, colon_values in zip(psalm.half_verse_nodes, half_verses, strict=True):
            vectors[node] = atomic_histogram(colon_values, vocabulary)
    return vectors


def atomic_psalm_vectors(
    psalms: list[MorphologicalPsalm], feature: FeatureKey
) -> dict[int, np.ndarray]:
    """Psalm-broadcast atomic histogram: word-count-weighted pooling across every colon."""
    vocabulary = _VOCABULARY_BY_FEATURE[feature]
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        half_verses = _half_verses_for_feature(psalm, feature)
        flattened = tuple(value for colon_values in half_verses for value in colon_values)
        psalm_vector = unigram_histogram(flattened, index_of, dim)
        for node in psalm.half_verse_nodes:
            vectors[node] = psalm_vector
    return vectors


def sp_plus_feature_vectors(
    psalms: list[MorphologicalPsalm], feature: FeatureKey
) -> dict[int, np.ndarray]:
    """`[sp_unigram; atomic(feature)]` per colon node."""
    feature_vectors = atomic_vectors(psalms, feature)
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, colon_sp in zip(psalm.half_verse_nodes, psalm.half_verse_sp, strict=True):
            vectors[node] = np.concatenate([pos_unigram_histogram(colon_sp), feature_vectors[node]])
    return vectors


def sp_plus_feature_psalm_vectors(
    psalms: list[MorphologicalPsalm], feature: FeatureKey
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `[sp_unigram; atomic(feature)]`."""
    sp_vectors = sp_unigram_psalm_vectors(psalms)
    feature_vectors = atomic_psalm_vectors(psalms, feature)
    return {node: np.concatenate([sp_vectors[node], feature_vectors[node]]) for node in sp_vectors}


def full_morphology_vectors(psalms: list[MorphologicalPsalm]) -> dict[int, np.ndarray]:
    """[sp; gn; nu; ps; st; vs; vt; prs_gn; prs_nu; prs_ps] per colon, dim 77 (H4.4 baseline)."""
    per_feature = {feature: atomic_vectors(psalms, feature) for feature in _FULL_FEATURE_ORDER}
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, colon_sp in zip(psalm.half_verse_nodes, psalm.half_verse_sp, strict=True):
            blocks = [pos_unigram_histogram(colon_sp)]
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
