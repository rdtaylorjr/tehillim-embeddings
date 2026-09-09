"""Phrase-signature histograms: bundle inventory, and cumulative bigram/trigram sequences."""

from __future__ import annotations

import numpy as np

from core.columns import PsalmColumns
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
from syntactic.corpus import PhrasePsalm
from syntactic.signature import psalm_signatures


def collapsed_signatures(
    psalm: PhrasePsalm, external_counts: dict[str, int], k: int
) -> tuple[tuple[str, ...], ...]:
    """One psalm's signature sequences, sub-threshold signatures collapsed to RARE."""
    return tuple(
        tuple(collapse_rare(signature, external_counts, k) for signature in half_verse)
        for half_verse in psalm_signatures(psalm)
    )


def _psalm_columns(
    psalms: list[PhrasePsalm], external_counts: dict[str, int], k: int
) -> list[PsalmColumns]:
    """Per-psalm (nodes, collapsed signature sequences) pairs, as the pooled builders take."""
    return [
        PsalmColumns(
            psalm.number, psalm.half_verse_nodes, collapsed_signatures(psalm, external_counts, k)
        )
        for psalm in psalms
    ]


def phrase_signature_vectors(
    psalms: list[PhrasePsalm], vocabulary: tuple[str, ...], external_counts: dict[str, int], k: int
) -> dict[int, np.ndarray]:
    """Phrase-signature inventory histogram (M_S), RARE-collapsed at the unigram level."""
    index_of = index_map(vocabulary)
    dim = len(vocabulary)
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        collapsed = collapsed_signatures(psalm, external_counts, k)
        for node, half_verse_sigs in zip(psalm.half_verse_nodes, collapsed, strict=True):
            vectors[node] = unigram_histogram(half_verse_sigs, index_of, dim)
    return vectors


def phrase_signature_psalm_vectors(
    psalms: list[PhrasePsalm], vocabulary: tuple[str, ...], external_counts: dict[str, int], k: int
) -> dict[int, np.ndarray]:
    """Psalm-broadcast phrase-signature inventory histogram, atom-count-weighted pooling."""
    columns = _psalm_columns(psalms, external_counts, k)
    return pooled_ngram_psalm_vectors(columns, (1,), vocabulary, order_by_node=None)


def phrase_signature_1_2gram_vectors(
    psalms: list[PhrasePsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """`[M_S; signature_bigram]`, over the RARE-collapsed signature vocabulary."""
    index_of = index_map(vocabulary)
    dim = len(vocabulary)
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        collapsed = collapsed_signatures(psalm, external_counts, k)
        for node, half_verse_sigs in zip(psalm.half_verse_nodes, collapsed, strict=True):
            ordered = reorder(half_verse_sigs, node, order_by_node)
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
    """Dense `[M_S; bigram; trigram]` per half-verse: the sparse path's exactness reference."""
    index_of = index_map(vocabulary)
    dim = len(vocabulary)
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        collapsed = collapsed_signatures(psalm, external_counts, k)
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


def phrase_signature_1_2gram_psalm_vectors(
    psalms: list[PhrasePsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `[M_S; signature_bigram]`, atom-count-weighted pooling."""
    columns = _psalm_columns(psalms, external_counts, k)
    return pooled_ngram_psalm_vectors(columns, (1, 2), vocabulary, order_by_node)


def phrase_signature_1_2_3gram_psalm_vectors(
    psalms: list[PhrasePsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Dense psalm-broadcast `[M_S; bigram; trigram]`: the sparse path's exactness reference."""
    columns = _psalm_columns(psalms, external_counts, k)
    return pooled_ngram_psalm_vectors(columns, (1, 2, 3), vocabulary, order_by_node)


def phrase_signature_1_2_3gram_sparse_vectors(
    psalms: list[PhrasePsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Sparse `[M_S; signature_bigram; signature_trigram]`: (indices, values)."""
    index_of = index_map(vocabulary)
    dim = len(vocabulary)
    vectors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for psalm in psalms:
        collapsed = collapsed_signatures(psalm, external_counts, k)
        for node, half_verse_sigs in zip(psalm.half_verse_nodes, collapsed, strict=True):
            ordered = reorder(half_verse_sigs, node, order_by_node)
            vectors[node] = sparse_1_2_3gram(ordered, index_of, dim)
    return vectors


def phrase_signature_1_2_3gram_psalm_sparse_vectors(
    psalms: list[PhrasePsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Psalm-broadcast sparse `[M_S; signature_bigram; signature_trigram]`, atom-count-weighted."""
    columns = _psalm_columns(psalms, external_counts, k)
    return sparse_pooled_1_2_3gram(columns, vocabulary, order_by_node)


#: One table both the production generator and the shuffle-null control resolve a builder from.
DENSE_BUILDERS = {
    "inventory": phrase_signature_vectors,
    "inventory_psalm": phrase_signature_psalm_vectors,
    "1_2gram": phrase_signature_1_2gram_vectors,
    "1_2gram_psalm": phrase_signature_1_2gram_psalm_vectors,
}

#: The trigram block is overwhelmingly zero at this dimension, so these are stored sparsely.
SPARSE_BUILDERS = {
    "1_2_3gram": phrase_signature_1_2_3gram_sparse_vectors,
    "1_2_3gram_psalm": phrase_signature_1_2_3gram_psalm_sparse_vectors,
}


#: The inventory histograms count without reading order, so a half-verse shuffle cannot move them.
ORDER_INVARIANT = ("inventory", "inventory_psalm")

#: The subset a shuffle-null control permutes: every builder here takes an order_by_node argument.
ORDERED_DENSE_BUILDERS = {
    "1_2gram": phrase_signature_1_2gram_vectors,
    "1_2gram_psalm": phrase_signature_1_2gram_psalm_vectors,
}
