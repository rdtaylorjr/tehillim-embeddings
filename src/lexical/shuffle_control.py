"""Deterministic within-psalm colon-order permutations, for a shuffle-null order-effect control."""

from __future__ import annotations

import numpy as np

from lexical.corpus import LexicalPsalm


def shuffled_order_by_psalm(psalms: list[LexicalPsalm], seed: int) -> dict[int, np.ndarray]:
    """One fixed-seed random permutation of colon indices per psalm, independent across psalms."""
    order_by_psalm: dict[int, np.ndarray] = {}
    for psalm in psalms:
        n = len(psalm.colon_nodes)
        rng = np.random.default_rng((psalm.number, seed))
        order_by_psalm[psalm.number] = rng.permutation(n)
    return order_by_psalm
