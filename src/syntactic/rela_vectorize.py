"""Normalized safe-relation unigram histogram per half-verse: `Para` masked before any counting."""

from __future__ import annotations

import numpy as np

from core.columns import PsalmColumns
from core.ngram import pooled_ngram_psalm_vectors, unigram_histogram
from syntactic.corpus import PhrasePsalm
from syntactic.rela import SAFE_RELA_VOCABULARY, half_verse_safe_rela

_INDEX_OF = {value: i for i, value in enumerate(SAFE_RELA_VOCABULARY)}
_DIM = len(SAFE_RELA_VOCABULARY)


def phrase_rela_unigram_histogram(half_verse_rela: tuple[str, ...]) -> np.ndarray:
    """Normalized safe-relation proportions over one half-verse, `Para` masked to `NA` first."""
    return unigram_histogram(half_verse_safe_rela(half_verse_rela), _INDEX_OF, _DIM)


def phrase_rela_1gram_vectors(psalms: list[PhrasePsalm]) -> dict[int, np.ndarray]:
    """One `phrase_rela_1gram` histogram per half-verse node."""
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, half_verse_rela in zip(
            psalm.half_verse_nodes, psalm.half_verse_rela, strict=True
        ):
            vectors[node] = phrase_rela_unigram_histogram(half_verse_rela)
    return vectors


def phrase_rela_1gram_psalm_vectors(psalms: list[PhrasePsalm]) -> dict[int, np.ndarray]:
    """Psalm-broadcast `phrase_rela_1gram`: atom-count-weighted pooling, `Para` masked first."""
    columns = [
        PsalmColumns(
            psalm.number,
            psalm.half_verse_nodes,
            tuple(half_verse_safe_rela(c) for c in psalm.half_verse_rela),
        )
        for psalm in psalms
    ]
    return pooled_ngram_psalm_vectors(columns, (1,), SAFE_RELA_VOCABULARY, order_by_node=None)
