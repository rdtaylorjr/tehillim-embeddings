"""Per-feature morphology histograms: normalized proportions, NA as part of the distribution."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Literal

import numpy as np

from core.ngram import (
    ngram_psalm_vectors,
    ngram_vectors,
    sparse_ngram_psalm_vectors,
    sparse_ngram_vectors,
    unigram_histogram,
)
from core.vocabulary import index_map
from morphological.corpus import MorphologicalPsalm
from morphological.vocabulary import (
    GN_VOCABULARY,
    NU_VOCABULARY,
    PRS_GN_VOCABULARY,
    PRS_NU_VOCABULARY,
    PRS_PS_VOCABULARY,
    PS_VOCABULARY,
    SP_VOCABULARY,
    ST_VOCABULARY,
    VS_VOCABULARY,
    VT_VOCABULARY,
    sp_columns,
)

_SP_INDEX = index_map(SP_VOCABULARY)

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


def feature_1gram_vectors(
    psalms: list[MorphologicalPsalm], feature: FeatureKey
) -> dict[int, np.ndarray]:
    """One atomic histogram per half-verse node for `feature`."""
    vocabulary = _VOCABULARY_BY_FEATURE[feature]
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        half_verses = half_verses_for_feature(psalm, feature)
        for node, half_verse_values in zip(psalm.half_verse_nodes, half_verses, strict=True):
            vectors[node] = atomic_histogram(half_verse_values, vocabulary)
    return vectors


def feature_1gram_psalm_vectors(
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
    feature_vectors = feature_1gram_vectors(psalms, feature)
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, half_verse_sp in zip(psalm.half_verse_nodes, psalm.half_verse_sp, strict=True):
            vectors[node] = np.concatenate(
                [
                    unigram_histogram(half_verse_sp, _SP_INDEX, len(SP_VOCABULARY)),
                    feature_vectors[node],
                ]
            )
    return vectors


def sp_plus_feature_psalm_vectors(
    psalms: list[MorphologicalPsalm], feature: FeatureKey
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `[sp_unigram; atomic(feature)]`."""
    sp_vectors = ngram_psalm_vectors(psalms, sp_columns, SP_VOCABULARY, (1,))
    feature_vectors = feature_1gram_psalm_vectors(psalms, feature)
    return {node: np.concatenate([sp_vectors[node], feature_vectors[node]]) for node in sp_vectors}


def concatenated_feature_vectors(
    psalms: list[MorphologicalPsalm], features: tuple[FeatureKey, ...]
) -> dict[int, np.ndarray]:
    """`[sp; ...features]` per half-verse node, in the order the features are given."""
    per_feature = {feature: feature_1gram_vectors(psalms, feature) for feature in features}
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, half_verse_sp in zip(psalm.half_verse_nodes, psalm.half_verse_sp, strict=True):
            blocks = [unigram_histogram(half_verse_sp, _SP_INDEX, len(SP_VOCABULARY))]
            blocks.extend(per_feature[feature][node] for feature in features)
            vectors[node] = np.concatenate(blocks)
    return vectors


def concatenated_feature_psalm_vectors(
    psalms: list[MorphologicalPsalm], features: tuple[FeatureKey, ...]
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `concatenated_feature_vectors`."""
    sp_vectors = ngram_psalm_vectors(psalms, sp_columns, SP_VOCABULARY, (1,))
    per_feature = {feature: feature_1gram_psalm_vectors(psalms, feature) for feature in features}
    vectors: dict[int, np.ndarray] = {}
    for node in sp_vectors:
        blocks = [sp_vectors[node]]
        blocks.extend(per_feature[feature][node] for feature in features)
        vectors[node] = np.concatenate(blocks)
    return vectors


full_morphology_vectors = partial(concatenated_feature_vectors, features=_FULL_FEATURE_ORDER)
full_morphology_psalm_vectors = partial(
    concatenated_feature_psalm_vectors, features=_FULL_FEATURE_ORDER
)


def feature_columns(feature: FeatureKey, psalm: MorphologicalPsalm) -> tuple[tuple[str, ...], ...]:
    """One feature's value sequence per half-verse, which its n-grams read."""
    return half_verses_for_feature(psalm, feature)


#: The vocabularies large enough that a dense trigram block would dwarf the corpus it describes.
SPARSE_TRIGRAM_FEATURES: frozenset[FeatureKey] = frozenset({"vs"})


def feature_1_2gram_vectors(
    psalms: list[MorphologicalPsalm],
    feature: FeatureKey,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """`[atomic; bigram]` per half-verse node for `feature`."""
    return ngram_vectors(
        psalms,
        partial(feature_columns, feature),
        _VOCABULARY_BY_FEATURE[feature],
        (1, 2),
        order_by_node,
    )


def feature_1_2gram_psalm_vectors(
    psalms: list[MorphologicalPsalm],
    feature: FeatureKey,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `[atomic; bigram]` for `feature`."""
    return ngram_psalm_vectors(
        psalms,
        partial(feature_columns, feature),
        _VOCABULARY_BY_FEATURE[feature],
        (1, 2),
        order_by_node,
    )


def feature_1_2_3gram_vectors(
    psalms: list[MorphologicalPsalm],
    feature: FeatureKey,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """`[atomic; bigram; trigram]` per half-verse node for `feature`."""
    return ngram_vectors(
        psalms,
        partial(feature_columns, feature),
        _VOCABULARY_BY_FEATURE[feature],
        (1, 2, 3),
        order_by_node,
    )


def feature_1_2_3gram_psalm_vectors(
    psalms: list[MorphologicalPsalm],
    feature: FeatureKey,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `[atomic; bigram; trigram]` for `feature`."""
    return ngram_psalm_vectors(
        psalms,
        partial(feature_columns, feature),
        _VOCABULARY_BY_FEATURE[feature],
        (1, 2, 3),
        order_by_node,
    )


def feature_sparse_trigram_vectors(
    psalms: list[MorphologicalPsalm],
    feature: FeatureKey,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Sparse `[atomic; bigram; trigram]` for a feature whose dense block would be unwieldy."""
    return sparse_ngram_vectors(
        psalms,
        partial(feature_columns, feature),
        _VOCABULARY_BY_FEATURE[feature],
        order_by_node,
    )


def feature_sparse_trigram_psalm_vectors(
    psalms: list[MorphologicalPsalm],
    feature: FeatureKey,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Psalm-broadcast sparse `[atomic; bigram; trigram]` for `feature`."""
    return sparse_ngram_psalm_vectors(
        psalms,
        partial(feature_columns, feature),
        _VOCABULARY_BY_FEATURE[feature],
        order_by_node,
    )


def vocabulary_for_feature(feature: FeatureKey) -> tuple[str, ...]:
    """The frozen vocabulary one atomic feature's histograms are counted over."""
    return _VOCABULARY_BY_FEATURE[feature]
