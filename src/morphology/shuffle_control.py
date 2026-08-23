"""Order-shuffle-null primitive: word order within a colon, not colon order within a psalm."""

from __future__ import annotations

import numpy as np

from morphology.corpus import MorphologicalPsalm


def shuffled_within_colon_order(
    psalms: list[MorphologicalPsalm], seed: int
) -> dict[int, np.ndarray]:
    """One fixed-seed permutation of word indices per colon node, keyed by (node, seed)."""
    order_by_node: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, colon_sp in zip(psalm.half_verse_nodes, psalm.half_verse_sp, strict=True):
            m = len(colon_sp)
            rng = np.random.default_rng((node, seed))
            order_by_node[node] = rng.permutation(m)
    return order_by_node
