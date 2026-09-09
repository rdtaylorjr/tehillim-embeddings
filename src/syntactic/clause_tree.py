"""Conventional shape summaries of a psalm's clause-atom dependency forest, stage 6D."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from syntactic.clause_edge import distance_bin

if TYPE_CHECKING:
    from syntactic.corpus import ClausePsalm

__all__ = ["TREE_SUMMARY_FIELDS", "clause_tree_summary", "mother_positions"]

#: Psalm-level only: 65 percent of colons hold fewer than two clause atoms and have no tree.
TREE_SUMMARY_FIELDS: tuple[str, ...] = (
    "n_atoms",
    "n_roots",
    "mean_depth",
    "max_depth",
    "mean_branching",
    "max_branching",
    "adjacent_link_proportion",
    "long_range_link_proportion",
    "leaf_proportion",
)


def mother_positions(psalm: ClausePsalm) -> np.ndarray:
    """Each clause atom's mother as a position, or -1 where it roots the forest."""
    index_of = {node: position for position, node in enumerate(psalm.clause_atom_nodes)}
    return np.array(
        [
            index_of.get(mother, -1) if mother is not None else -1
            for mother in psalm.clause_atom_mother
        ],
        dtype=np.int64,
    )


def _depths(parent: np.ndarray) -> np.ndarray:
    """Depth of every node above its root, relaxed in lockstep rather than walked per chain."""
    depth = np.zeros(parent.size, dtype=np.int64)
    has_parent = parent >= 0
    safe_parent = np.where(has_parent, parent, 0)
    for _ in range(parent.size):
        updated = np.where(has_parent, depth[safe_parent] + 1, 0)
        if np.array_equal(updated, depth):
            return depth
        depth = updated
    raise ValueError("clause-atom mother links did not settle, so the forest holds a cycle")


def clause_tree_summary(psalm: ClausePsalm) -> np.ndarray:
    """One psalm's dependency-forest shape, in `TREE_SUMMARY_FIELDS` order."""
    parent = mother_positions(psalm)
    n_atoms = parent.size
    if n_atoms == 0:
        return np.zeros(len(TREE_SUMMARY_FIELDS), dtype=np.float32)
    has_parent = parent >= 0
    depth = _depths(parent)
    daughters = np.bincount(parent[has_parent], minlength=n_atoms)
    spans = np.abs(np.arange(n_atoms)[has_parent] - parent[has_parent])
    n_edges = int(has_parent.sum())
    bins = [distance_bin(int(span)) for span in spans]
    return np.array(
        [
            n_atoms,
            int((~has_parent).sum()),
            float(depth.mean()),
            float(depth.max()),
            float(daughters.mean()),
            float(daughters.max()),
            bins.count("1") / n_edges if n_edges else 0.0,
            bins.count("5+") / n_edges if n_edges else 0.0,
            float((daughters == 0).mean()),
        ],
        dtype=np.float32,
    )


def clause_tree_summary_psalm_vectors(psalms: list[ClausePsalm]) -> dict[int, np.ndarray]:
    """Psalm-broadcast tree summary: every colon of a psalm carries its psalm's forest shape."""
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        summary = clause_tree_summary(psalm)
        for node in psalm.half_verse_nodes:
            vectors[node] = summary
    return vectors
