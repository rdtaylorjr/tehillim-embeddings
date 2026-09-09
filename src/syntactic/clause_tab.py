"""BHSA hierarchical clause-atom depth and its within-colon contour, stage 6F."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from core.columns import PsalmColumns
from core.ngram import pooled_ngram_psalm_vectors, reorder
from syntactic.assignment import majority_mask
from syntactic.clause_columns import colon_sequences
from syntactic.clause_ngram import dense_ngram_psalm_vectors, dense_ngram_vectors

if TYPE_CHECKING:
    from syntactic.corpus import ClausePsalm

__all__ = [
    "TAB_TRANSITION_VOCABULARY",
    "TAB_VOCABULARY",
    "clause_tab_columns",
    "clause_tab_depth_columns",
    "clause_tab_inventory_psalm_vectors",
    "clause_tab_inventory_vectors",
    "clause_tab_transition_columns",
    "clause_tab_transition_psalm_vectors",
    "tab_label",
    "tab_transition_label",
]

#: Capped at 10 because BHSA reaches 29 whole-Bible, and depths above 10 are 7.1% of Psalms.
_TAB_CAP = 10

TAB_VOCABULARY: tuple[str, ...] = (
    *(str(depth) for depth in range(_TAB_CAP + 1)),
    f"{_TAB_CAP + 1}+",
)

#: Signed and capped both ways, so a rare deep jump cannot open a near-empty dimension.
TAB_TRANSITION_VOCABULARY: tuple[str, ...] = (
    "<=-3",
    "-2",
    "-1",
    "0",
    "+1",
    "+2",
    "+3",
    "+4",
    ">=+5",
)


def tab_label(depth: int) -> str:
    """One clause atom's hierarchical depth, capped so the thin tail stays a single dimension."""
    return str(depth) if depth <= _TAB_CAP else f"{_TAB_CAP + 1}+"


def tab_transition_label(change: int) -> str:
    """One step of the depth contour, capped in both directions."""
    if change <= -3:
        return "<=-3"
    if change >= 5:
        return ">=+5"
    return f"{change:+d}" if change else "0"


def _depth_columns(psalm: ClausePsalm) -> tuple[tuple[str, ...], ...]:
    """Uncapped per-colon depth sequences as strings, so a contour step measures the real change."""
    assignment = psalm.clause_atom_assignment
    return colon_sequences(
        tuple(str(depth) for depth in psalm.clause_atom_tab),
        assignment,
        len(psalm.half_verse_nodes),
        mask=majority_mask(assignment),
    )


def clause_tab_columns(psalm: ClausePsalm) -> tuple[tuple[str, ...], ...]:
    """One capped-depth sequence per colon, under the majority rule."""
    return tuple(
        tuple(tab_label(int(depth)) for depth in column) for column in _depth_columns(psalm)
    )


def clause_tab_transition_columns(
    psalm: ClausePsalm, order_by_node: dict[int, np.ndarray] | None = None
) -> tuple[tuple[str, ...], ...]:
    """One depth-contour sequence per colon, holding one step fewer than the colon's atoms."""
    #: The permutation applies to the depths, since permuting differences reorders no real text.
    return tuple(
        tuple(
            tab_transition_label(int(later) - int(earlier))
            for earlier, later in pairwise(reorder(column, node, order_by_node))
        )
        for node, column in zip(psalm.half_verse_nodes, _depth_columns(psalm), strict=True)
    )


def clause_tab_inventory_vectors(psalms: list[ClausePsalm]) -> dict[int, np.ndarray]:
    """One `clause_tab_inventory` histogram per colon node."""
    return dense_ngram_vectors(psalms, clause_tab_columns, TAB_VOCABULARY, (1,))


def clause_tab_inventory_psalm_vectors(psalms: list[ClausePsalm]) -> dict[int, np.ndarray]:
    """Psalm-broadcast `clause_tab_inventory`."""
    return dense_ngram_psalm_vectors(psalms, clause_tab_columns, TAB_VOCABULARY, (1,))


def clause_tab_transition_psalm_vectors(
    psalms: list[ClausePsalm], order_by_node: dict[int, np.ndarray] | None = None
) -> dict[int, np.ndarray]:
    """Psalm-broadcast depth contour: colon level is omitted, 65 percent of colons have no step."""
    columns = [
        PsalmColumns(
            psalm.number,
            psalm.half_verse_nodes,
            clause_tab_transition_columns(psalm, order_by_node),
        )
        for psalm in psalms
    ]
    return pooled_ngram_psalm_vectors(columns, (1,), TAB_TRANSITION_VOCABULARY, order_by_node=None)


def clause_tab_depth_columns(psalm: ClausePsalm) -> tuple[tuple[str, ...], ...]:
    """Uncapped depth sequences, the length the shuffle permutation is built over."""
    return _depth_columns(psalm)
