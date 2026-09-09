"""Normalized clause-kind inventory per colon: the coarsest clause representation, stage 6A."""

from __future__ import annotations

import numpy as np

from core.columns import PsalmColumns
from core.ngram import pooled_ngram_psalm_vectors, unigram_histogram
from syntactic.assignment import majority_mask
from syntactic.clause_columns import colon_sequences
from syntactic.clause_vocabulary import KIND_VOCABULARY
from syntactic.corpus import ClausePsalm

__all__ = [
    "clause_kind_columns",
    "clause_kind_inventory_psalm_vectors",
    "clause_kind_inventory_vectors",
]

_INDEX_OF = {value: index for index, value in enumerate(KIND_VOCABULARY)}
_DIM = len(KIND_VOCABULARY)


def clause_kind_columns(psalm: ClausePsalm) -> tuple[tuple[str, ...], ...]:
    """One clause-kind sequence per colon, under the majority assignment rule."""
    return colon_sequences(
        psalm.clause_kind,
        psalm.clause_assignment,
        len(psalm.half_verse_nodes),
        mask=majority_mask(psalm.clause_assignment),
    )


def clause_kind_inventory_vectors(psalms: list[ClausePsalm]) -> dict[int, np.ndarray]:
    """One `clause_kind_inventory` histogram per colon node."""
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, kinds in zip(psalm.half_verse_nodes, clause_kind_columns(psalm), strict=True):
            vectors[node] = unigram_histogram(kinds, _INDEX_OF, _DIM)
    return vectors


def clause_kind_inventory_psalm_vectors(psalms: list[ClausePsalm]) -> dict[int, np.ndarray]:
    """Psalm-broadcast `clause_kind_inventory`: clause-count-weighted pooling across the psalm."""
    columns = [
        PsalmColumns(psalm.number, psalm.half_verse_nodes, clause_kind_columns(psalm))
        for psalm in psalms
    ]
    return pooled_ngram_psalm_vectors(columns, (1,), KIND_VOCABULARY, order_by_node=None)
