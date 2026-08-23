"""Phrase-signature histograms: bundle inventory, and cumulative bigram/trigram sequences."""

from __future__ import annotations

import numpy as np

from morphological.ngram import (
    bigram_histogram,
    pooled_ngram_psalm_vectors,
    reorder,
    trigram_histogram,
    unigram_histogram,
)
from phrase.corpus import PhrasePsalm
from phrase.signature import psalm_signatures
from phrase.signature_support import collapse_rare


def _collapsed_signatures(
    psalm: PhrasePsalm, external_counts: dict[str, int], k: int
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(collapse_rare(signature, external_counts, k) for signature in colon)
        for colon in psalm_signatures(psalm)
    )


def phrase_signature_vectors(
    psalms: list[PhrasePsalm], vocabulary: tuple[str, ...], external_counts: dict[str, int], k: int
) -> dict[int, np.ndarray]:
    """Phrase-signature inventory histogram (M_S), RARE-collapsed at the unigram level."""
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        collapsed = _collapsed_signatures(psalm, external_counts, k)
        for node, colon_sigs in zip(psalm.half_verse_nodes, collapsed, strict=True):
            vectors[node] = unigram_histogram(colon_sigs, index_of, dim)
    return vectors


def phrase_signature_psalm_vectors(
    psalms: list[PhrasePsalm], vocabulary: tuple[str, ...], external_counts: dict[str, int], k: int
) -> dict[int, np.ndarray]:
    """Psalm-broadcast phrase-signature inventory histogram, atom-count-weighted pooling."""
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)
    columns = [
        (psalm.half_verse_nodes, _collapsed_signatures(psalm, external_counts, k))
        for psalm in psalms
    ]
    return pooled_ngram_psalm_vectors(columns, (1,), index_of, dim, order_by_node=None)


def phrase_signature_1_2gram_vectors(
    psalms: list[PhrasePsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """`[M_S; signature_bigram]` per colon node, over the RARE-collapsed signature vocabulary."""
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


def phrase_signature_1_2_3gram_vectors(
    psalms: list[PhrasePsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """`[M_S; signature_bigram; signature_trigram]` per colon node."""
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


def phrase_signature_1_2gram_psalm_vectors(
    psalms: list[PhrasePsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `[M_S; signature_bigram]`, atom-count-weighted pooling."""
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)
    columns = [
        (psalm.half_verse_nodes, _collapsed_signatures(psalm, external_counts, k))
        for psalm in psalms
    ]
    return pooled_ngram_psalm_vectors(columns, (1, 2), index_of, dim, order_by_node)


def phrase_signature_1_2_3gram_psalm_vectors(
    psalms: list[PhrasePsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `[M_S; signature_bigram; signature_trigram]`, atom-count-weighted pooling."""
    index_of = {value: i for i, value in enumerate(vocabulary)}
    dim = len(vocabulary)
    columns = [
        (psalm.half_verse_nodes, _collapsed_signatures(psalm, external_counts, k))
        for psalm in psalms
    ]
    return pooled_ngram_psalm_vectors(columns, (1, 2, 3), index_of, dim, order_by_node)
