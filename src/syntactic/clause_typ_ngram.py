"""Clause-atom type unigram/bigram/trigram histograms over the RARE-collapsed vocabulary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from core.ngram import (
    ngram_psalm_vectors,
    ngram_vectors,
    sparse_ngram_psalm_vectors,
    sparse_ngram_vectors,
)
from syntactic.clause_ngram import ColumnsOf, collapsed_columns

if TYPE_CHECKING:
    from syntactic.corpus import ClausePsalm

__all__ = ["DENSE_BUILDERS", "SPARSE_BUILDERS", "clause_typ_columns"]


def clause_typ_columns(
    psalm: ClausePsalm,
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> tuple[tuple[str, ...], ...]:
    """One RARE-collapsed clause-atom type sequence per colon, under the majority rule."""
    return collapsed_columns(
        psalm.clause_atom_typ,
        psalm.clause_atom_assignment,
        len(psalm.half_verse_nodes),
        external_counts,
        k,
        order_by_node=order_by_node,
        nodes=psalm.half_verse_nodes,
    )


def _columns_of(external_counts: dict[str, int], k: int) -> ColumnsOf:
    """Binds the support table into the per-psalm column builder the vectorizers take."""
    return lambda psalm: clause_typ_columns(psalm, external_counts, k)


def clause_typ_1gram_vectors(
    psalms: list[ClausePsalm], vocabulary: tuple[str, ...], external_counts: dict[str, int], k: int
) -> dict[int, np.ndarray]:
    """One `clause_typ_1gram` histogram per colon node."""
    return ngram_vectors(psalms, _columns_of(external_counts, k), vocabulary, (1,))


def clause_typ_1_2gram_vectors(
    psalms: list[ClausePsalm], vocabulary: tuple[str, ...], external_counts: dict[str, int], k: int
) -> dict[int, np.ndarray]:
    """`[clause_typ_1gram; clause_typ_bigram]` per colon node."""
    return ngram_vectors(psalms, _columns_of(external_counts, k), vocabulary, (1, 2))


def clause_typ_1gram_psalm_vectors(
    psalms: list[ClausePsalm], vocabulary: tuple[str, ...], external_counts: dict[str, int], k: int
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `clause_typ_1gram`."""
    return ngram_psalm_vectors(psalms, _columns_of(external_counts, k), vocabulary, (1,))


def clause_typ_1_2gram_psalm_vectors(
    psalms: list[ClausePsalm], vocabulary: tuple[str, ...], external_counts: dict[str, int], k: int
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `[clause_typ_1gram; clause_typ_bigram]`."""
    return ngram_psalm_vectors(psalms, _columns_of(external_counts, k), vocabulary, (1, 2))


def clause_typ_1_2_3gram_sparse_vectors(
    psalms: list[ClausePsalm], vocabulary: tuple[str, ...], external_counts: dict[str, int], k: int
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Sparse `[unigram; bigram; trigram]` per colon node."""
    return sparse_ngram_vectors(psalms, _columns_of(external_counts, k), vocabulary)


def clause_typ_1_2_3gram_psalm_sparse_vectors(
    psalms: list[ClausePsalm], vocabulary: tuple[str, ...], external_counts: dict[str, int], k: int
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Psalm-broadcast sparse `[unigram; bigram; trigram]`."""
    return sparse_ngram_psalm_vectors(psalms, _columns_of(external_counts, k), vocabulary)


DENSE_BUILDERS = {
    "1gram": clause_typ_1gram_vectors,
    "1_2gram": clause_typ_1_2gram_vectors,
    "1gram_psalm": clause_typ_1gram_psalm_vectors,
    "1_2gram_psalm": clause_typ_1_2gram_psalm_vectors,
}

#: The trigram block is overwhelmingly zero at this dimension, so it is stored sparsely.
SPARSE_BUILDERS = {
    "1_2_3gram": clause_typ_1_2_3gram_sparse_vectors,
    "1_2_3gram_psalm": clause_typ_1_2_3gram_psalm_sparse_vectors,
}
