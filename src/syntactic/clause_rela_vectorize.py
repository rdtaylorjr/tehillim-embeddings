"""Clause-relation and joint clause-signature histograms, with the parallelism firewall applied."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from core.ngram import (
    ngram_psalm_vectors,
    ngram_vectors,
    sparse_ngram_psalm_vectors,
    sparse_ngram_vectors,
)
from syntactic.clause_code import Scope
from syntactic.clause_ngram import ColumnsOf, collapsed_columns
from syntactic.clause_rela import retained_rela_indices
from syntactic.clause_signature import clause_signatures

if TYPE_CHECKING:
    from syntactic.corpus import ClausePsalm

__all__ = [
    "SIGNATURE_DENSE_BUILDERS",
    "SIGNATURE_SPARSE_BUILDERS",
    "clause_rela_1gram_psalm_vectors",
    "clause_rela_1gram_vectors",
    "safe_clause_mask",
]


def safe_clause_mask(psalm: ClausePsalm) -> np.ndarray:
    """Which clauses parallelism scope may see, excluding the audited resumption relations."""
    keep = np.zeros(len(psalm.clause_nodes), dtype=bool)
    keep[list(retained_rela_indices(psalm.clause_rela, Scope.PARALLELISM))] = True
    return keep


def clause_rela_columns(
    psalm: ClausePsalm,
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> tuple[tuple[str, ...], ...]:
    """One RARE-collapsed clause-relation sequence per colon, firewall applied."""
    return collapsed_columns(
        psalm.clause_rela,
        psalm.clause_assignment,
        len(psalm.half_verse_nodes),
        external_counts,
        k,
        safe_clause_mask(psalm),
        order_by_node=order_by_node,
        nodes=psalm.half_verse_nodes,
    )


def clause_signature_columns(
    psalm: ClausePsalm,
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> tuple[tuple[str, ...], ...]:
    """One RARE-collapsed `typ:rela` sequence per colon, firewall applied."""
    return collapsed_columns(
        clause_signatures(typ=psalm.clause_typ, rela=psalm.clause_rela),
        psalm.clause_assignment,
        len(psalm.half_verse_nodes),
        external_counts,
        k,
        safe_clause_mask(psalm),
        order_by_node=order_by_node,
        nodes=psalm.half_verse_nodes,
    )


def _rela_columns_of(external_counts: dict[str, int], k: int) -> ColumnsOf:
    """Binds the support table into the per-psalm relation column builder."""
    return lambda psalm: clause_rela_columns(psalm, external_counts, k)


def _signature_columns_of(external_counts: dict[str, int], k: int) -> ColumnsOf:
    """Binds the support table into the per-psalm signature column builder."""
    return lambda psalm: clause_signature_columns(psalm, external_counts, k)


def clause_rela_1gram_vectors(
    psalms: list[ClausePsalm], vocabulary: tuple[str, ...], external_counts: dict[str, int], k: int
) -> dict[int, np.ndarray]:
    """One `clause_rela_1gram` histogram per colon node."""
    return ngram_vectors(psalms, _rela_columns_of(external_counts, k), vocabulary, (1,))


def clause_rela_1gram_psalm_vectors(
    psalms: list[ClausePsalm], vocabulary: tuple[str, ...], external_counts: dict[str, int], k: int
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `clause_rela_1gram`."""
    return ngram_psalm_vectors(psalms, _rela_columns_of(external_counts, k), vocabulary, (1,))


def clause_signature_1gram_vectors(
    psalms: list[ClausePsalm], vocabulary: tuple[str, ...], external_counts: dict[str, int], k: int
) -> dict[int, np.ndarray]:
    """One `clause_signature` unigram histogram per colon node."""
    return ngram_vectors(psalms, _signature_columns_of(external_counts, k), vocabulary, (1,))


def clause_signature_1gram_psalm_vectors(
    psalms: list[ClausePsalm], vocabulary: tuple[str, ...], external_counts: dict[str, int], k: int
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `clause_signature` unigram histogram."""
    return ngram_psalm_vectors(psalms, _signature_columns_of(external_counts, k), vocabulary, (1,))


def clause_signature_1_2gram_vectors(
    psalms: list[ClausePsalm], vocabulary: tuple[str, ...], external_counts: dict[str, int], k: int
) -> dict[int, np.ndarray]:
    """`[clause_signature_1gram; clause_signature_bigram]` per colon node."""
    return ngram_vectors(psalms, _signature_columns_of(external_counts, k), vocabulary, (1, 2))


def clause_signature_1_2gram_psalm_vectors(
    psalms: list[ClausePsalm], vocabulary: tuple[str, ...], external_counts: dict[str, int], k: int
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `[clause_signature_1gram; clause_signature_bigram]`."""
    return ngram_psalm_vectors(
        psalms, _signature_columns_of(external_counts, k), vocabulary, (1, 2)
    )


def clause_signature_1_2_3gram_sparse_vectors(
    psalms: list[ClausePsalm], vocabulary: tuple[str, ...], external_counts: dict[str, int], k: int
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Sparse `[unigram; bigram; trigram]` clause signatures per colon node."""
    return sparse_ngram_vectors(psalms, _signature_columns_of(external_counts, k), vocabulary)


def clause_signature_1_2_3gram_psalm_sparse_vectors(
    psalms: list[ClausePsalm], vocabulary: tuple[str, ...], external_counts: dict[str, int], k: int
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Psalm-broadcast sparse `[unigram; bigram; trigram]` clause signatures."""
    return sparse_ngram_psalm_vectors(psalms, _signature_columns_of(external_counts, k), vocabulary)


SIGNATURE_DENSE_BUILDERS = {
    "1gram": clause_signature_1gram_vectors,
    "1_2gram": clause_signature_1_2gram_vectors,
    "1gram_psalm": clause_signature_1gram_psalm_vectors,
    "1_2gram_psalm": clause_signature_1_2gram_psalm_vectors,
}

#: The trigram block is overwhelmingly zero at this dimension, so it is stored sparsely.
SIGNATURE_SPARSE_BUILDERS = {
    "1_2_3gram": clause_signature_1_2_3gram_sparse_vectors,
    "1_2_3gram_psalm": clause_signature_1_2_3gram_psalm_sparse_vectors,
}


def clause_rela_1_2gram_vectors(
    psalms: list[ClausePsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """`[clause_rela_1gram; clause_rela_bigram]` per colon node."""
    return ngram_vectors(
        psalms, _rela_columns_of(external_counts, k), vocabulary, (1, 2), order_by_node
    )


def clause_rela_1_2gram_psalm_vectors(
    psalms: list[ClausePsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `[clause_rela_1gram; clause_rela_bigram]`."""
    return ngram_psalm_vectors(
        psalms, _rela_columns_of(external_counts, k), vocabulary, (1, 2), order_by_node
    )


def clause_rela_1_2_3gram_vectors(
    psalms: list[ClausePsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """`[clause_rela_1gram; bigram; trigram]` per colon node, dense at this vocabulary size."""
    return ngram_vectors(
        psalms, _rela_columns_of(external_counts, k), vocabulary, (1, 2, 3), order_by_node
    )


def clause_rela_1_2_3gram_psalm_vectors(
    psalms: list[ClausePsalm],
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    order_by_node: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `[clause_rela_1gram; bigram; trigram]`."""
    return ngram_psalm_vectors(
        psalms, _rela_columns_of(external_counts, k), vocabulary, (1, 2, 3), order_by_node
    )
