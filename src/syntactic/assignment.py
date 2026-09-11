"""Assigns syntactic nodes to the half verses they overlap, by word fraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Assignment", "assign_nodes_to_units", "containment_mask", "majority_mask"]

#: A weight is one word-count ratio, so exact equality with 1.0 would reject whole nodes.
_CONTAINMENT_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class Assignment:
    """Every node-unit overlap as parallel arrays, ordered by node then unit."""

    node_index: np.ndarray
    unit_index: np.ndarray
    weight: np.ndarray


def assign_nodes_to_units(
    slot_unit: np.ndarray, slots: np.ndarray, counts: np.ndarray, *, n_units: int
) -> Assignment:
    """Overlap weights of each node against each unit, as the fraction of the node's words in it."""
    if counts.size and int(counts.min()) <= 0:
        raise ValueError("a node with no slots has no word fraction to assign")
    node_of_slot = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    unit_of_slot = slot_unit[slots]
    inside = unit_of_slot >= 0
    #: Packing the pair into one integer lets a single sorted unique do the whole group-by.
    packed = node_of_slot[inside] * n_units + unit_of_slot[inside]
    pairs, overlap = np.unique(packed, return_counts=True)
    node_index = pairs // n_units
    return Assignment(
        node_index=node_index,
        unit_index=pairs % n_units,
        weight=overlap / counts[node_index],
    )


def containment_mask(assignment: Assignment) -> np.ndarray:
    """Selects the overlaps whose node lies wholly inside one unit."""
    return assignment.weight >= 1.0 - _CONTAINMENT_TOLERANCE


def majority_mask(assignment: Assignment) -> np.ndarray:
    """Selects one overlap per node, the unit holding most of its words, earliest unit on a tie."""
    #: Sorting by node, then descending weight, then unit puts each node's winner first.
    order = np.lexsort((assignment.unit_index, -assignment.weight, assignment.node_index))
    ranked_nodes = assignment.node_index[order]
    winners = np.ones(ranked_nodes.size, dtype=bool)
    winners[1:] = ranked_nodes[1:] != ranked_nodes[:-1]
    mask = np.zeros(ranked_nodes.size, dtype=bool)
    mask[order[winners]] = True
    return mask
