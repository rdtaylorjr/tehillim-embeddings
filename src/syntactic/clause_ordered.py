"""Order-sensitive clause families, resolved to everything one shuffle draw needs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from core.support import build_signature_vocabulary, load_external_signature_counts
from syntactic.clause_ngram import (
    ColumnsOf,
    dense_ngram_psalm_vectors,
    dense_ngram_vectors,
    sparse_1_2_3gram_psalm_vectors,
    sparse_1_2_3gram_vectors,
)
from syntactic.clause_rela_vectorize import clause_signature_columns
from syntactic.clause_support import (
    MIN_EXTERNAL_SUPPORT_K_CLAUSE_ATOM_TYP,
    MIN_EXTERNAL_SUPPORT_K_CLAUSE_SIGNATURE,
)
from syntactic.clause_tab import TAB_TRANSITION_VOCABULARY, clause_tab_transition_columns
from syntactic.clause_typ_ngram import clause_typ_columns

if TYPE_CHECKING:
    from collections.abc import Mapping

    from syntactic.corpus import ClausePsalm

__all__ = ["ORDERED_UNITS", "OrderedFamily", "resolve_family"]

#: A construction's n-gram orders, and whether it pools across the psalm, read from its name.
_DENSE_ORDERS: Mapping[str, tuple[int, ...]] = {
    "1_2gram": (1, 2),
    "1_2gram_psalm": (1, 2),
    "transition_psalm": (1,),
}
_SPARSE_CONSTRUCTIONS = ("1_2_3gram", "1_2_3gram_psalm")

#: Each order-sensitive family: its support table and frozen k, or a closed vocabulary instead.
_TABLE_DRIVEN = {
    "typ": (
        "clause_atom_typ_external_support.csv",
        MIN_EXTERNAL_SUPPORT_K_CLAUSE_ATOM_TYP,
        clause_typ_columns,
    ),
    "signature": (
        "clause_signature_external_support.csv",
        MIN_EXTERNAL_SUPPORT_K_CLAUSE_SIGNATURE,
        clause_signature_columns,
    ),
}

_CLOSED = {
    "tab": (TAB_TRANSITION_VOCABULARY, clause_tab_transition_columns),
}

ORDERED_UNITS: tuple[str, ...] = (*sorted(_TABLE_DRIVEN), *sorted(_CLOSED))


@dataclass(frozen=True, slots=True)
class OrderedFamily:
    """One clause family whose vectors depend on within-colon order."""

    unit: str
    vocabulary: tuple[str, ...]
    columns_of: ColumnsOf
    dense: tuple[str, ...]
    sparse: tuple[str, ...]


def resolve_family(unit: str, config_root: Path) -> OrderedFamily:
    """Binds a family's frozen vocabulary and column builder, whichever source it declares."""
    if unit in _TABLE_DRIVEN:
        filename, k, columns = _TABLE_DRIVEN[unit]
        counts = load_external_signature_counts(config_root / filename)
        return OrderedFamily(
            unit=unit,
            vocabulary=build_signature_vocabulary(counts, k),
            columns_of=partial(columns, external_counts=counts, k=k),
            dense=("1_2gram", "1_2gram_psalm"),
            sparse=_SPARSE_CONSTRUCTIONS,
        )
    vocabulary, columns_of = _CLOSED[unit]
    return OrderedFamily(
        unit=unit,
        vocabulary=vocabulary,
        columns_of=columns_of,
        dense=("transition_psalm",),
        sparse=(),
    )


def dense_vectors(
    family: OrderedFamily,
    psalms: list[ClausePsalm],
    construction: str,
    order_by_node: dict[int, np.ndarray] | None,
) -> dict[int, np.ndarray]:
    """One dense construction's vectors, pooled across the psalm when its name says so."""
    build = dense_ngram_psalm_vectors if construction.endswith("_psalm") else dense_ngram_vectors
    return build(
        psalms, family.columns_of, family.vocabulary, _DENSE_ORDERS[construction], order_by_node
    )


def sparse_vectors(
    family: OrderedFamily,
    psalms: list[ClausePsalm],
    construction: str,
    order_by_node: dict[int, np.ndarray] | None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """One sparse construction's vectors, pooled across the psalm when its name says so."""
    build = (
        sparse_1_2_3gram_psalm_vectors
        if construction.endswith("_psalm")
        else sparse_1_2_3gram_vectors
    )
    return build(psalms, family.columns_of, family.vocabulary, order_by_node)
