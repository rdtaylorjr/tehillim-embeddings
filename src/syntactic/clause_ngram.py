"""Shared clause n-gram vectorizing: rare-collapsed half-verse sequences into dense or sparse."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from core.ngram import permuted_columns
from core.support import collapse_rare
from syntactic.assignment import Assignment, majority_mask
from syntactic.clause_columns import half_verse_sequences

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


__all__ = ["ColumnsOf", "collapsed_columns"]

type ColumnsOf = Callable[..., tuple[tuple[str, ...], ...]]


def collapsed_columns(
    values: Sequence[str],
    assignment: Assignment,
    n_half_verses: int,
    external_counts: dict[str, int],
    k: int,
    keep: np.ndarray | None = None,
    order_by_node: dict[int, np.ndarray] | None = None,
    nodes: tuple[int, ...] = (),
) -> tuple[tuple[str, ...], ...]:
    """Per-half-verse value sequences under the majority rule, rare values collapsed first."""
    mask = majority_mask(assignment)
    if keep is not None:
        #: Excluded nodes leave through the mask, so the surviving indices still address `values`.
        mask &= keep[assignment.node_index]
    collapsed = tuple(collapse_rare(value, external_counts, k) for value in values)
    columns = half_verse_sequences(collapsed, assignment, n_half_verses, mask=mask)
    return permuted_columns(columns, nodes, order_by_node)
