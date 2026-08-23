"""Order-shuffle-null primitive: phrase-atom order within a colon, not colon order in a psalm."""

from __future__ import annotations

import numpy as np

from phrase.corpus import PhrasePsalm


def shuffled_within_colon_order(psalms: list[PhrasePsalm], seed: int) -> dict[int, np.ndarray]:
    """One fixed-seed permutation of phrase-atom indices per colon node, keyed by (node, seed)."""
    order_by_node: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, colon_typ in zip(psalm.half_verse_nodes, psalm.half_verse_typ, strict=True):
            m = len(colon_typ)
            rng = np.random.default_rng((node, seed))
            order_by_node[node] = rng.permutation(m)
    return order_by_node
