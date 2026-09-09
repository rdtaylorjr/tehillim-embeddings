"""Independent-marginals baseline `[clause_typ; clause_rela]`, the joint signature's control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from syntactic.clause_ngram import (
    collapsed_columns,
    dense_ngram_psalm_vectors,
    dense_ngram_vectors,
)
from syntactic.clause_rela_vectorize import clause_rela_columns, safe_clause_mask

if TYPE_CHECKING:
    from syntactic.corpus import ClausePsalm

__all__ = ["MarginalSide", "clause_marginal_psalm_vectors", "clause_marginal_vectors"]


@dataclass(frozen=True, slots=True)
class MarginalSide:
    """One half of the marginal: its frozen vocabulary and the support table behind it."""

    vocabulary: tuple[str, ...]
    external_counts: dict[str, int]
    k: int


def clause_typ_at_clause_level_columns(
    psalm: ClausePsalm, external_counts: dict[str, int], k: int
) -> tuple[tuple[str, ...], ...]:
    """Clause-level type sequences per colon, under the same firewall the relation side uses."""
    return collapsed_columns(
        psalm.clause_typ,
        psalm.clause_assignment,
        len(psalm.half_verse_nodes),
        external_counts,
        k,
        safe_clause_mask(psalm),
    )


def clause_marginal_vectors(
    psalms: list[ClausePsalm], typ: MarginalSide, rela: MarginalSide
) -> dict[int, np.ndarray]:
    """`[clause_typ_1gram; clause_rela_1gram]` per colon node, each side normalized separately."""
    typ_vectors = dense_ngram_vectors(
        psalms,
        lambda psalm: clause_typ_at_clause_level_columns(psalm, typ.external_counts, typ.k),
        typ.vocabulary,
        (1,),
    )
    rela_vectors = dense_ngram_vectors(
        psalms,
        lambda psalm: clause_rela_columns(psalm, rela.external_counts, rela.k),
        rela.vocabulary,
        (1,),
    )
    return {node: np.concatenate([typ_vectors[node], rela_vectors[node]]) for node in typ_vectors}


def clause_marginal_psalm_vectors(
    psalms: list[ClausePsalm], typ: MarginalSide, rela: MarginalSide
) -> dict[int, np.ndarray]:
    """Psalm-broadcast `[clause_typ_1gram; clause_rela_1gram]`."""
    typ_vectors = dense_ngram_psalm_vectors(
        psalms,
        lambda psalm: clause_typ_at_clause_level_columns(psalm, typ.external_counts, typ.k),
        typ.vocabulary,
        (1,),
    )
    rela_vectors = dense_ngram_psalm_vectors(
        psalms,
        lambda psalm: clause_rela_columns(psalm, rela.external_counts, rela.k),
        rela.vocabulary,
        (1,),
    )
    return {node: np.concatenate([typ_vectors[node], rela_vectors[node]]) for node in typ_vectors}
