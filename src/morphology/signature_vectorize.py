"""Grammatical-signature histograms: bundle inventory, and cumulative bigram/trigram sequences."""

from __future__ import annotations

import numpy as np

from morphology.atomic import (
    FeatureKey,
    _half_verses_for_feature,
    atomic_histogram,
    atomic_psalm_vectors,
)
from morphology.corpus import MorphologicalPsalm
from morphology.ngram import (
    bigram_histogram,
    pooled_ngram_psalm_vectors,
    reorder,
    sparse_1_2_3gram,
    sparse_pooled_1_2_3gram,
    trigram_histogram,
    unigram_histogram,
)
from morphology.pos_ngram import pos_unigram_histogram, sp_unigram_psalm_vectors
from morphology.signature import psalm_signatures
from morphology.signature_support import collapse_rare
from morphology.vocabulary import (
    GN_VOCABULARY,
    NU_VOCABULARY,
    PS_VOCABULARY,
    ST_VOCABULARY,
    VS_VOCABULARY,
    VT_VOCABULARY,
)

_CORE_FEATURE_ORDER: tuple[FeatureKey, ...] = ("gn", "nu", "ps", "st", "vs", "vt")
_CORE_VOCABULARY: dict[FeatureKey, tuple[str, ...]] = {
    "gn": GN_VOCABULARY,
    "nu": NU_VOCABULARY,
    "ps": PS_VOCABULARY,
    "st": ST_VOCABULARY,
    "vs": VS_VOCABULARY,
    "vt": VT_VOCABULARY,
}


def morph_atomic_vectors(psalms: list[MorphologicalPsalm]) -> dict[int, np.ndarray]:
    """`[sp; gn; nu; ps; st; vs; vt]` per colon node, dim 66 (4C.1's baseline)."""
    per_feature = {
        feature: atomic_histogram_column(psalms, feature) for feature in _CORE_FEATURE_ORDER
    }
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, colon_sp in zip(psalm.half_verse_nodes, psalm.half_verse_sp, strict=True):
            blocks = [pos_unigram_histogram(colon_sp)]
            blocks.extend(per_feature[feature][node] for feature in _CORE_FEATURE_ORDER)
            vectors[node] = np.concatenate(blocks)
    return vectors


def atomic_histogram_column(
    psalms: list[MorphologicalPsalm], feature: FeatureKey
) -> dict[int, np.ndarray]:
    """One atomic histogram per colon node for `feature`, reusing `atomic.atomic_histogram`."""
    vocabulary = _CORE_VOCABULARY[feature]
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        half_verses = _half_verses_for_feature(psalm, feature)
        for node, colon_values in zip(psalm.half_verse_nodes, half_verses, strict=True):
            vectors[node] = atomic_histogram(colon_values, vocabulary)
    return vectors


def morph_atomic_psalm_vectors(psalms: list[MorphologicalPsalm]) -> dict[int, np.ndarray]:
    """Psalm-broadcast `morph_atomic_vectors`."""
    sp_vectors = sp_unigram_psalm_vectors(psalms)
    per_feature = {
        feature: atomic_psalm_vectors(psalms, feature) for feature in _CORE_FEATURE_ORDER
    }
    vectors: dict[int, np.ndarray] = {}
    for node in sp_vectors:
        blocks = [sp_vectors[node]]
        blocks.extend(per_feature[feature][node] for feature in _CORE_FEATURE_ORDER)
        vectors[node] = np.concatenate(blocks)
    return vectors


def _collapsed_signatures(
    psalm: MorphologicalPsalm, external_counts: dict[str, int], k: int
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(collapse_rare(signature, external_counts, k) for signature in colon)
        for colon in psalm_signatures(psalm)
    )


def morph_signature_vectors(
    psalms: list[MorphologicalPsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
) -> dict[int, np.ndarray]:
    """`morph_signature` inventory histogram (M_G), RARE-collapsed at the unigram level."""
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        collapsed = _collapsed_signatures(psalm, external_counts, k)
        for node, colon_sigs in zip(psalm.half_verse_nodes, collapsed, strict=True):
            vectors[node] = unigram_histogram(colon_sigs, index_of, dim)
    return vectors


def morph_signature_psalm_vectors(
    psalms: list[MorphologicalPsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `morph_signature` inventory histogram, word-count-weighted pooling."""
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)
    columns = [
        (psalm.half_verse_nodes, _collapsed_signatures(psalm, external_counts, k))
        for psalm in psalms
    ]
    return pooled_ngram_psalm_vectors(columns, (1,), index_of, dim, order_by_node=None)


def morph_signature_1_2gram_vectors(
    psalms: list[MorphologicalPsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """`[M_G; signature_bigram]` per colon node, over the RARE-collapsed signature vocabulary."""
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        collapsed = _collapsed_signatures(psalm, external_counts, k)
        for node, colon_sigs in zip(psalm.half_verse_nodes, collapsed, strict=True):
            ordered = reorder(colon_sigs, node, order_by_node)
            vectors[node] = np.concatenate(
                [
                    unigram_histogram(ordered, index_of, dim),
                    bigram_histogram(ordered, index_of, dim),
                ]
            )
    return vectors


def morph_signature_1_2_3gram_vectors(
    psalms: list[MorphologicalPsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """`[M_G; signature_bigram; signature_trigram]` per colon node."""
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        collapsed = _collapsed_signatures(psalm, external_counts, k)
        for node, colon_sigs in zip(psalm.half_verse_nodes, collapsed, strict=True):
            ordered = reorder(colon_sigs, node, order_by_node)
            vectors[node] = np.concatenate(
                [
                    unigram_histogram(ordered, index_of, dim),
                    bigram_histogram(ordered, index_of, dim),
                    trigram_histogram(ordered, index_of, dim),
                ]
            )
    return vectors


def morph_signature_1_2gram_psalm_vectors(
    psalms: list[MorphologicalPsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `[M_G; signature_bigram]`, word-count-weighted pooling."""
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)
    columns = [
        (psalm.half_verse_nodes, _collapsed_signatures(psalm, external_counts, k))
        for psalm in psalms
    ]
    return pooled_ngram_psalm_vectors(columns, (1, 2), index_of, dim, order_by_node)


def morph_signature_1_2_3gram_psalm_vectors(
    psalms: list[MorphologicalPsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `[M_G; signature_bigram; signature_trigram]`, word-count-weighted pooling."""
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)
    columns = [
        (psalm.half_verse_nodes, _collapsed_signatures(psalm, external_counts, k))
        for psalm in psalms
    ]
    return pooled_ngram_psalm_vectors(columns, (1, 2, 3), index_of, dim, order_by_node)


def morph_signature_1_2_3gram_sparse_vectors(
    psalms: list[MorphologicalPsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Sparse `[M_G; signature_bigram; signature_trigram]` per colon node: (indices, values)."""
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)
    vectors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for psalm in psalms:
        collapsed = _collapsed_signatures(psalm, external_counts, k)
        for node, colon_sigs in zip(psalm.half_verse_nodes, collapsed, strict=True):
            ordered = reorder(colon_sigs, node, order_by_node)
            vectors[node] = sparse_1_2_3gram(ordered, index_of, dim)
    return vectors


def morph_signature_1_2_3gram_psalm_sparse_vectors(
    psalms: list[MorphologicalPsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Psalm-broadcast sparse `[M_G; signature_bigram; signature_trigram]`, word-count-weighted."""
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)
    columns = [
        (psalm.half_verse_nodes, _collapsed_signatures(psalm, external_counts, k))
        for psalm in psalms
    ]
    return sparse_pooled_1_2_3gram(columns, index_of, dim, order_by_node)
