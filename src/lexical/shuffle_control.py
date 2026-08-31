"""Deterministic within-psalm colon-order permutations, for a shuffle-null order-effect control."""

from __future__ import annotations

import numpy as np

from lexical.corpus import LexicalPsalm

# 1000 puts the p-value floor at 1/1001, under the 0.05/7 needed for BH across genres.
DEFAULT_N_SHUFFLES = 1000


def shuffled_order_by_psalm(psalms: list[LexicalPsalm], seed: int) -> dict[int, np.ndarray]:
    """One fixed-seed random permutation of colon indices per psalm, independent across psalms."""
    order_by_psalm: dict[int, np.ndarray] = {}
    for psalm in psalms:
        n = len(psalm.colon_nodes)
        rng = np.random.default_rng((psalm.number, seed))
        order_by_psalm[psalm.number] = rng.permutation(n)
    return order_by_psalm
