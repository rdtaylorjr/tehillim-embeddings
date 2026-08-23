"""Normalized safe subphrase-relation histogram per colon: `par` masked before any counting."""

from __future__ import annotations

import numpy as np

from morphology.ngram import pooled_ngram_psalm_vectors, unigram_histogram
from syntax.corpus import PhrasePsalm
from syntax.subphrase import SAFE_SUBPHRASE_RELA_VOCABULARY, colon_safe_subphrase_rela

_INDEX_OF = {value: i for i, value in enumerate(SAFE_SUBPHRASE_RELA_VOCABULARY)}
_DIM = len(SAFE_SUBPHRASE_RELA_VOCABULARY)


def subphrase_rela_unigram_histogram(colon_rela: tuple[str, ...]) -> np.ndarray:
    """Normalized safe subphrase-relation proportions, `par` masked to `NA` first."""
    return unigram_histogram(colon_safe_subphrase_rela(colon_rela), _INDEX_OF, _DIM)


def subphrase_rela_1gram_vectors(psalms: list[PhrasePsalm]) -> dict[int, np.ndarray]:
    """One `subphrase_rela_1gram` histogram per colon node."""
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, colon_rela in zip(psalm.colon_nodes, psalm.colon_subphrase_rela, strict=True):
            vectors[node] = subphrase_rela_unigram_histogram(colon_rela)
    return vectors


def subphrase_rela_1gram_psalm_vectors(psalms: list[PhrasePsalm]) -> dict[int, np.ndarray]:
    """Psalm-broadcast `subphrase_rela_1gram`: subphrase-count-weighted pooling, `par` masked."""
    columns = [
        (
            psalm.colon_nodes,
            tuple(colon_safe_subphrase_rela(c) for c in psalm.colon_subphrase_rela),
        )
        for psalm in psalms
    ]
    return pooled_ngram_psalm_vectors(columns, (1,), _INDEX_OF, _DIM, order_by_node=None)
