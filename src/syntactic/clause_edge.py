"""Clause dependency-span representation over BHSA `mother` links between clause atoms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from core.ngram import ngram_psalm_vectors, ngram_vectors
from syntactic.assignment import majority_mask
from syntactic.clause_columns import colon_sequences

if TYPE_CHECKING:
    from syntactic.corpus import ClausePsalm

__all__ = [
    "EDGE_DISTANCE_VOCABULARY",
    "NO_MOTHER",
    "clause_edge_distance_columns",
    "clause_edge_distance_psalm_vectors",
    "clause_edge_distance_vectors",
    "distance_bin",
    "edge_reaches_outside_colon",
]

#: Marks a daughter whose mother lies outside the psalm's extracted clause atoms.
NO_MOTHER = "<ROOT>"

#: Four spans and a root marker, closed and structural, so no support threshold applies to it.
EDGE_DISTANCE_VOCABULARY: tuple[str, ...] = ("1", "2", "3-4", "5+", NO_MOTHER)


def distance_bin(distance: int) -> str:
    """The frozen conservative bin for one edge's clause-atom span, sign ignored."""
    span = abs(distance)
    if span == 1:
        return "1"
    if span == 2:
        return "2"
    if span <= 4:
        return "3-4"
    return "5+"


def _mother_position(psalm: ClausePsalm) -> list[int | None]:
    """Each clause atom's mother as a position, or None where it roots the forest."""
    index_of = {node: position for position, node in enumerate(psalm.clause_atom_nodes)}
    return [
        index_of.get(mother) if mother is not None else None for mother in psalm.clause_atom_mother
    ]


def clause_edge_distance_columns(psalm: ClausePsalm) -> tuple[tuple[str, ...], ...]:
    """One dependency-span sequence per colon, keyed to the daughter's colon."""
    mothers = _mother_position(psalm)
    values = tuple(
        NO_MOTHER if mother is None else distance_bin(position - mother)
        for position, mother in enumerate(mothers)
    )
    assignment = psalm.clause_atom_assignment
    return colon_sequences(
        values, assignment, len(psalm.half_verse_nodes), mask=majority_mask(assignment)
    )


def edge_reaches_outside_colon(psalm: ClausePsalm) -> np.ndarray:
    """Which clause atoms depend on a mother sitting in a different colon."""
    assignment = psalm.clause_atom_assignment
    mask = majority_mask(assignment)
    colon_of = dict(
        zip(assignment.node_index[mask].tolist(), assignment.unit_index[mask].tolist(), strict=True)
    )
    outside = np.zeros(len(psalm.clause_atom_nodes), dtype=bool)
    for position, mother in enumerate(_mother_position(psalm)):
        if mother is None:
            continue
        outside[position] = colon_of.get(position) != colon_of.get(mother)
    return outside


def clause_edge_distance_vectors(psalms: list[ClausePsalm]) -> dict[int, np.ndarray]:
    """One `edge_distance` histogram per colon node, over the closed span vocabulary."""
    return ngram_vectors(psalms, clause_edge_distance_columns, EDGE_DISTANCE_VOCABULARY, (1,))


def clause_edge_distance_psalm_vectors(psalms: list[ClausePsalm]) -> dict[int, np.ndarray]:
    """Psalm-broadcast `edge_distance`, pooled across the psalm's colons."""
    return ngram_psalm_vectors(psalms, clause_edge_distance_columns, EDGE_DISTANCE_VOCABULARY, (1,))
