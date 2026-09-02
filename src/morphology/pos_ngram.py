"""Normalized POS unigram/bigram/trigram histograms per half-verse, concatenated cumulatively."""

from __future__ import annotations

import numpy as np

from core.ngram import (
    bigram_histogram,
    pooled_ngram_psalm_vectors,
    reorder,
    trigram_histogram,
    unigram_histogram,
)
from morphology.corpus import MorphologicalPsalm
from morphology.vocabulary import SP_VOCABULARY

_INDEX_OF = {value: i for i, value in enumerate(SP_VOCABULARY)}
_DIM = len(SP_VOCABULARY)


def pos_unigram_histogram(half_verse_sp: tuple[str, ...]) -> np.ndarray:
    """Normalized POS-tag proportions over one half-verse: count(tag) / m."""
    return unigram_histogram(half_verse_sp, _INDEX_OF, _DIM)


def pos_bigram_histogram(half_verse_sp: tuple[str, ...]) -> np.ndarray:
    """Normalized adjacent-POS-tag-pair proportions over one half-verse: count(pair) / (m - 1)."""
    return bigram_histogram(half_verse_sp, _INDEX_OF, _DIM)


def pos_trigram_histogram(half_verse_sp: tuple[str, ...]) -> np.ndarray:
    """Normalized POS-tag-triple proportions over one half-verse: count(triple) / (m - 2)."""
    return trigram_histogram(half_verse_sp, _INDEX_OF, _DIM)


def sp_unigram_vectors(psalms: list[MorphologicalPsalm]) -> dict[int, np.ndarray]:
    """One `sp_unigram` histogram per half-verse node."""
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, half_verse_sp in zip(psalm.half_verse_nodes, psalm.half_verse_sp, strict=True):
            vectors[node] = pos_unigram_histogram(half_verse_sp)
    return vectors


def sp_1_2gram_vectors(
    psalms: list[MorphologicalPsalm], order_by_node: dict[int, np.ndarray] | None = None
) -> dict[int, np.ndarray]:
    """`[sp_unigram; sp_bigram]`, words optionally reordered within their half-verse."""
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, half_verse_sp in zip(psalm.half_verse_nodes, psalm.half_verse_sp, strict=True):
            ordered = reorder(half_verse_sp, node, order_by_node)
            vectors[node] = np.concatenate(
                [pos_unigram_histogram(ordered), pos_bigram_histogram(ordered)]
            )
    return vectors


def sp_1_2_3gram_vectors(
    psalms: list[MorphologicalPsalm], order_by_node: dict[int, np.ndarray] | None = None
) -> dict[int, np.ndarray]:
    """`[sp_unigram; sp_bigram; sp_trigram]` per half-verse node."""
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, half_verse_sp in zip(psalm.half_verse_nodes, psalm.half_verse_sp, strict=True):
            ordered = reorder(half_verse_sp, node, order_by_node)
            vectors[node] = np.concatenate(
                [
                    pos_unigram_histogram(ordered),
                    pos_bigram_histogram(ordered),
                    pos_trigram_histogram(ordered),
                ]
            )
    return vectors


def sp_unigram_psalm_vectors(psalms: list[MorphologicalPsalm]) -> dict[int, np.ndarray]:
    """Psalm-broadcast `sp_unigram`: word-count-weighted pooling across every half-verse."""
    columns = [(p.half_verse_nodes, p.half_verse_sp) for p in psalms]
    return pooled_ngram_psalm_vectors(columns, (1,), _INDEX_OF, _DIM, order_by_node=None)


def sp_1_2gram_psalm_vectors(
    psalms: list[MorphologicalPsalm], order_by_node: dict[int, np.ndarray] | None = None
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `[sp_unigram; sp_bigram]`, word-count-weighted pooling."""
    columns = [(p.half_verse_nodes, p.half_verse_sp) for p in psalms]
    return pooled_ngram_psalm_vectors(columns, (1, 2), _INDEX_OF, _DIM, order_by_node)


def sp_1_2_3gram_psalm_vectors(
    psalms: list[MorphologicalPsalm], order_by_node: dict[int, np.ndarray] | None = None
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `[sp_unigram; sp_bigram; sp_trigram]`, word-count-weighted pooling."""
    columns = [(p.half_verse_nodes, p.half_verse_sp) for p in psalms]
    return pooled_ngram_psalm_vectors(columns, (1, 2, 3), _INDEX_OF, _DIM, order_by_node)
