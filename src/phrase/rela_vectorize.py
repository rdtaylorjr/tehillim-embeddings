"""Normalized safe-relation unigram histogram per colon: `Para` masked before any counting."""

from __future__ import annotations

import numpy as np

from morphological.ngram import pooled_ngram_psalm_vectors, unigram_histogram
from phrase.corpus import PhrasePsalm
from phrase.rela import SAFE_RELA_VOCABULARY, colon_safe_rela

_INDEX_OF = {value: i for i, value in enumerate(SAFE_RELA_VOCABULARY)}
_DIM = len(SAFE_RELA_VOCABULARY)


def phrase_rela_unigram_histogram(colon_rela: tuple[str, ...]) -> np.ndarray:
    """Normalized safe-relation proportions over one colon, `Para` masked to `NA` first."""
    return unigram_histogram(colon_safe_rela(colon_rela), _INDEX_OF, _DIM)


def phrase_rela_1gram_vectors(psalms: list[PhrasePsalm]) -> dict[int, np.ndarray]:
    """One `phrase_rela_1gram` histogram per colon node."""
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, colon_rela in zip(psalm.half_verse_nodes, psalm.half_verse_rela, strict=True):
            vectors[node] = phrase_rela_unigram_histogram(colon_rela)
    return vectors


def phrase_rela_1gram_psalm_vectors(psalms: list[PhrasePsalm]) -> dict[int, np.ndarray]:
    """Psalm-broadcast `phrase_rela_1gram`: atom-count-weighted pooling, `Para` masked first."""
    columns = [
        (psalm.half_verse_nodes, tuple(colon_safe_rela(c) for c in psalm.half_verse_rela))
        for psalm in psalms
    ]
    return pooled_ngram_psalm_vectors(columns, (1,), _INDEX_OF, _DIM, order_by_node=None)
