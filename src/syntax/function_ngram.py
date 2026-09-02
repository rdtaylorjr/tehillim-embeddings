"""Normalized phrase-function unigram/bigram/trigram histograms per node, cumulative concat."""

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
from syntax.corpus import PhrasePsalm
from syntax.vocabulary import FUNCTION_VOCABULARY

_INDEX_OF = {value: i for i, value in enumerate(FUNCTION_VOCABULARY)}
_DIM = len(FUNCTION_VOCABULARY)


def phrase_function_unigram_histogram(half_verse_function: tuple[str, ...]) -> np.ndarray:
    """Normalized phrase-function proportions over one half-verse: count(function) / m."""
    return unigram_histogram(half_verse_function, _INDEX_OF, _DIM)


def phrase_function_bigram_histogram(half_verse_function: tuple[str, ...]) -> np.ndarray:
    """Normalized adjacent-function-pair proportions over one half-verse: count(pair) / (m - 1)."""
    return bigram_histogram(half_verse_function, _INDEX_OF, _DIM)


def phrase_function_trigram_histogram(half_verse_function: tuple[str, ...]) -> np.ndarray:
    """Normalized phrase-function-triple proportions over one node: count(triple) / (m - 2)."""
    return trigram_histogram(half_verse_function, _INDEX_OF, _DIM)


def phrase_function_1gram_vectors(psalms: list[PhrasePsalm]) -> dict[int, np.ndarray]:
    """One `phrase_function_1gram` histogram per half-verse node."""
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, half_verse_function in zip(
            psalm.half_verse_nodes, psalm.half_verse_function, strict=True
        ):
            vectors[node] = phrase_function_unigram_histogram(half_verse_function)
    return vectors


def phrase_function_1_2gram_vectors(
    psalms: list[PhrasePsalm], order_by_node: dict[int, np.ndarray] | None = None
) -> dict[int, np.ndarray]:
    """`[phrase_function_1gram; phrase_function_bigram]` per half-verse node."""
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, half_verse_function in zip(
            psalm.half_verse_nodes, psalm.half_verse_function, strict=True
        ):
            ordered = reorder(half_verse_function, node, order_by_node)
            vectors[node] = np.concatenate(
                [
                    phrase_function_unigram_histogram(ordered),
                    phrase_function_bigram_histogram(ordered),
                ]
            )
    return vectors


def phrase_function_1_2_3gram_vectors(
    psalms: list[PhrasePsalm], order_by_node: dict[int, np.ndarray] | None = None
) -> dict[int, np.ndarray]:
    """`[phrase_function_1gram; bigram; trigram]` per half-verse node."""
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, half_verse_function in zip(
            psalm.half_verse_nodes, psalm.half_verse_function, strict=True
        ):
            ordered = reorder(half_verse_function, node, order_by_node)
            vectors[node] = np.concatenate(
                [
                    phrase_function_unigram_histogram(ordered),
                    phrase_function_bigram_histogram(ordered),
                    phrase_function_trigram_histogram(ordered),
                ]
            )
    return vectors


def phrase_function_1gram_psalm_vectors(psalms: list[PhrasePsalm]) -> dict[int, np.ndarray]:
    """Psalm-broadcast `phrase_function_1gram`: atom-count-weighted pooling."""
    columns = [PsalmColumns(p.number, p.half_verse_nodes, p.half_verse_function) for p in psalms]
    return pooled_ngram_psalm_vectors(columns, (1,), FUNCTION_VOCABULARY, order_by_node=None)


def phrase_function_1_2gram_psalm_vectors(
    psalms: list[PhrasePsalm], order_by_node: dict[int, np.ndarray] | None = None
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `[phrase_function_1gram; bigram]`, atom-count-weighted pooling."""
    columns = [PsalmColumns(p.number, p.half_verse_nodes, p.half_verse_function) for p in psalms]
    return pooled_ngram_psalm_vectors(columns, (1, 2), FUNCTION_VOCABULARY, order_by_node)


def phrase_function_1_2_3gram_psalm_vectors(
    psalms: list[PhrasePsalm], order_by_node: dict[int, np.ndarray] | None = None
) -> dict[int, np.ndarray]:
    """Dense psalm-broadcast `[1gram; bigram; trigram]`: the sparse path's exactness reference."""
    columns = [PsalmColumns(p.number, p.half_verse_nodes, p.half_verse_function) for p in psalms]
    return pooled_ngram_psalm_vectors(columns, (1, 2, 3), FUNCTION_VOCABULARY, order_by_node)


def phrase_function_1_2_3gram_sparse_vectors(
    psalms: list[PhrasePsalm], order_by_node: dict[int, np.ndarray] | None = None
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Sparse `[phrase_function_1gram; bigram; trigram]` per half-verse node: (indices, values)."""
    vectors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for psalm in psalms:
        for node, half_verse in zip(psalm.half_verse_nodes, psalm.half_verse_function, strict=True):
            ordered = reorder(half_verse, node, order_by_node)
            vectors[node] = sparse_1_2_3gram(ordered, _INDEX_OF, _DIM)
    return vectors


def phrase_function_1_2_3gram_psalm_sparse_vectors(
    psalms: list[PhrasePsalm], order_by_node: dict[int, np.ndarray] | None = None
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Psalm-broadcast sparse `[phrase_function_1gram; bigram; trigram]`, atom-count-weighted."""
    columns = [PsalmColumns(p.number, p.half_verse_nodes, p.half_verse_function) for p in psalms]
    return sparse_pooled_1_2_3gram(columns, FUNCTION_VOCABULARY, order_by_node)
