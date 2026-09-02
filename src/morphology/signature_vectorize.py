"""Grammatical-signature histograms: bundle inventory, and cumulative bigram/trigram sequences."""

from __future__ import annotations

import numpy as np

from core.ngram import (
    bigram_histogram,
    pooled_ngram_psalm_vectors,
    reorder,
    sparse_1_2_3gram,
    sparse_pooled_1_2_3gram,
    trigram_histogram,
    unigram_histogram,
)
from core.support import collapse_rare
from core.vocabulary import index_map
from morphology.atomic import FeatureKey, atomic_psalm_vectors, atomic_vectors
from morphology.corpus import MorphologicalPsalm
from morphology.pos_ngram import pos_unigram_histogram, sp_unigram_psalm_vectors
from morphology.signature import psalm_signatures

_CORE_FEATURE_ORDER: tuple[FeatureKey, ...] = ("gn", "nu", "ps", "st", "vs", "vt")


def morph_atomic_vectors(psalms: list[MorphologicalPsalm]) -> dict[int, np.ndarray]:
    """`[sp; gn; nu; ps; st; vs; vt]` per half-verse node, dim 66 (4C.1's baseline)."""
    per_feature = {feature: atomic_vectors(psalms, feature) for feature in _CORE_FEATURE_ORDER}
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, half_verse_sp in zip(psalm.half_verse_nodes, psalm.half_verse_sp, strict=True):
            blocks = [pos_unigram_histogram(half_verse_sp)]
            blocks.extend(per_feature[feature][node] for feature in _CORE_FEATURE_ORDER)
            vectors[node] = np.concatenate(blocks)
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


def _psalm_columns(
    psalms: list[MorphologicalPsalm], external_counts: dict[str, int], k: int
) -> list[tuple[tuple[int, ...], tuple[tuple[str, ...], ...]]]:
    """Per-psalm (nodes, collapsed signature sequences) pairs, as the pooled builders take."""
    return [
        (psalm.half_verse_nodes, _collapsed_signatures(psalm, external_counts, k))
        for psalm in psalms
    ]


def _collapsed_signatures(
    psalm: MorphologicalPsalm, external_counts: dict[str, int], k: int
) -> tuple[tuple[str, ...], ...]:
    """One psalm's signature sequences, sub-threshold signatures collapsed to RARE."""
    return tuple(
        tuple(collapse_rare(signature, external_counts, k) for signature in half_verse)
        for half_verse in psalm_signatures(psalm)
    )


def morph_signature_vectors(
    psalms: list[MorphologicalPsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
) -> dict[int, np.ndarray]:
    """`morph_signature` inventory histogram (M_G), RARE-collapsed at the unigram level."""
    index_of = index_map(vocabulary)
    dim = len(vocabulary)
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        collapsed = _collapsed_signatures(psalm, external_counts, k)
        for node, half_verse_sigs in zip(psalm.half_verse_nodes, collapsed, strict=True):
            vectors[node] = unigram_histogram(half_verse_sigs, index_of, dim)
    return vectors


def morph_signature_psalm_vectors(
    psalms: list[MorphologicalPsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `morph_signature` inventory histogram, word-count-weighted pooling."""
    index_of = index_map(vocabulary)
    dim = len(vocabulary)
    columns = _psalm_columns(psalms, external_counts, k)
    return pooled_ngram_psalm_vectors(columns, (1,), index_of, dim, order_by_node=None)


def morph_signature_1_2gram_vectors(
    psalms: list[MorphologicalPsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """`[M_G; signature_bigram]`, over the RARE-collapsed signature vocabulary."""
    index_of = index_map(vocabulary)
    dim = len(vocabulary)
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        collapsed = _collapsed_signatures(psalm, external_counts, k)
        for node, half_verse_sigs in zip(psalm.half_verse_nodes, collapsed, strict=True):
            ordered = reorder(half_verse_sigs, node, order_by_node)
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
    """Dense `[M_G; bigram; trigram]` per half-verse node: the sparse path's exactness reference."""
    index_of = index_map(vocabulary)
    dim = len(vocabulary)
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        collapsed = _collapsed_signatures(psalm, external_counts, k)
        for node, half_verse_sigs in zip(psalm.half_verse_nodes, collapsed, strict=True):
            ordered = reorder(half_verse_sigs, node, order_by_node)
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
    index_of = index_map(vocabulary)
    dim = len(vocabulary)
    columns = _psalm_columns(psalms, external_counts, k)
    return pooled_ngram_psalm_vectors(columns, (1, 2), index_of, dim, order_by_node)


def morph_signature_1_2_3gram_psalm_vectors(
    psalms: list[MorphologicalPsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Dense psalm-broadcast `[M_G; bigram; trigram]`: the sparse path's exactness reference."""
    index_of = index_map(vocabulary)
    dim = len(vocabulary)
    columns = _psalm_columns(psalms, external_counts, k)
    return pooled_ngram_psalm_vectors(columns, (1, 2, 3), index_of, dim, order_by_node)


def morph_signature_1_2_3gram_sparse_vectors(
    psalms: list[MorphologicalPsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Sparse `[M_G; signature_bigram; signature_trigram]`: (indices, values)."""
    index_of = index_map(vocabulary)
    dim = len(vocabulary)
    vectors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for psalm in psalms:
        collapsed = _collapsed_signatures(psalm, external_counts, k)
        for node, half_verse_sigs in zip(psalm.half_verse_nodes, collapsed, strict=True):
            ordered = reorder(half_verse_sigs, node, order_by_node)
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
    index_of = index_map(vocabulary)
    dim = len(vocabulary)
    columns = _psalm_columns(psalms, external_counts, k)
    return sparse_pooled_1_2_3gram(columns, index_of, dim, order_by_node)
