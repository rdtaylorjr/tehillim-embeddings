"""Turns a psalm's clause assignment into one ordered value sequence per half-verse."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from syntactic.assignment import Assignment

__all__ = ["half_verse_sequences"]


def half_verse_sequences(
    values: Sequence[str], assignment: Assignment, n_half_verses: int, *, mask: np.ndarray
) -> tuple[tuple[str, ...], ...]:
    """Each half-verse's selected node values, in corpus node order."""
    units = assignment.unit_index[mask]
    nodes = assignment.node_index[mask]
    #: Sorting by half-verse then node makes one split yield every sequence already ordered.
    order = np.lexsort((nodes, units))
    counts = np.bincount(units[order], minlength=n_half_verses)
    grouped = np.split(nodes[order], np.cumsum(counts)[:-1])
    return tuple(tuple(values[index] for index in group) for group in grouped)
