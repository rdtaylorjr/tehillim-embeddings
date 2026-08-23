"""Normalized phrase-type unigram/bigram/trigram histograms per colon, cumulative concatenation."""

from __future__ import annotations

import numpy as np

from morphology.ngram import (
    bigram_histogram,
    pooled_ngram_psalm_vectors,
    reorder,
    trigram_histogram,
    unigram_histogram,
)
from syntax.corpus import PhrasePsalm
from syntax.vocabulary import TYP_VOCABULARY

_INDEX_OF = {value: i for i, value in enumerate(TYP_VOCABULARY)}
_DIM = len(TYP_VOCABULARY)


def phrase_typ_unigram_histogram(colon_typ: tuple[str, ...]) -> np.ndarray:
    """Normalized phrase-type proportions over one colon: count(typ) / m."""
    return unigram_histogram(colon_typ, _INDEX_OF, _DIM)


def phrase_typ_bigram_histogram(colon_typ: tuple[str, ...]) -> np.ndarray:
    """Normalized adjacent-phrase-type-pair proportions over one colon: count(pair) / (m - 1)."""
    return bigram_histogram(colon_typ, _INDEX_OF, _DIM)


def phrase_typ_trigram_histogram(colon_typ: tuple[str, ...]) -> np.ndarray:
    """Normalized phrase-type-triple proportions over one colon: count(triple) / (m - 2)."""
    return trigram_histogram(colon_typ, _INDEX_OF, _DIM)


def phrase_typ_1gram_vectors(psalms: list[PhrasePsalm]) -> dict[int, np.ndarray]:
    """One `phrase_typ_1gram` histogram per colon node."""
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, colon_typ in zip(psalm.half_verse_nodes, psalm.half_verse_typ, strict=True):
            vectors[node] = phrase_typ_unigram_histogram(colon_typ)
    return vectors


def phrase_typ_1_2gram_vectors(
    psalms: list[PhrasePsalm], order_by_node: dict[int, np.ndarray] | None = None
) -> dict[int, np.ndarray]:
    """`[phrase_typ_1gram; phrase_typ_bigram]` per colon node."""
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, colon_typ in zip(psalm.half_verse_nodes, psalm.half_verse_typ, strict=True):
            ordered = reorder(colon_typ, node, order_by_node)
            vectors[node] = np.concatenate(
                [phrase_typ_unigram_histogram(ordered), phrase_typ_bigram_histogram(ordered)]
            )
    return vectors


def phrase_typ_1_2_3gram_vectors(
    psalms: list[PhrasePsalm], order_by_node: dict[int, np.ndarray] | None = None
) -> dict[int, np.ndarray]:
    """`[phrase_typ_1gram; phrase_typ_bigram; phrase_typ_trigram]` per colon node."""
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, colon_typ in zip(psalm.half_verse_nodes, psalm.half_verse_typ, strict=True):
            ordered = reorder(colon_typ, node, order_by_node)
            vectors[node] = np.concatenate(
                [
                    phrase_typ_unigram_histogram(ordered),
                    phrase_typ_bigram_histogram(ordered),
                    phrase_typ_trigram_histogram(ordered),
                ]
            )
    return vectors


def phrase_typ_1gram_psalm_vectors(psalms: list[PhrasePsalm]) -> dict[int, np.ndarray]:
    """Psalm-broadcast `phrase_typ_1gram`: atom-count-weighted pooling across every colon."""
    columns = [(p.half_verse_nodes, p.half_verse_typ) for p in psalms]
    return pooled_ngram_psalm_vectors(columns, (1,), _INDEX_OF, _DIM, order_by_node=None)


def phrase_typ_1_2gram_psalm_vectors(
    psalms: list[PhrasePsalm], order_by_node: dict[int, np.ndarray] | None = None
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `[phrase_typ_1gram; phrase_typ_bigram]`, atom-count-weighted pooling."""
    columns = [(p.half_verse_nodes, p.half_verse_typ) for p in psalms]
    return pooled_ngram_psalm_vectors(columns, (1, 2), _INDEX_OF, _DIM, order_by_node)


def phrase_typ_1_2_3gram_psalm_vectors(
    psalms: list[PhrasePsalm], order_by_node: dict[int, np.ndarray] | None = None
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `[phrase_typ_1gram; bigram; trigram]`, atom-count-weighted pooling."""
    columns = [(p.half_verse_nodes, p.half_verse_typ) for p in psalms]
    return pooled_ngram_psalm_vectors(columns, (1, 2, 3), _INDEX_OF, _DIM, order_by_node)
