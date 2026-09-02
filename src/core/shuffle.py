"""Deterministic order permutations and dataset naming for the shuffle-null order controls."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

#: 1000 puts the p-value floor at 1/1001, at or below 0.05/7, the smallest BH critical value.
DEFAULT_N_SHUFFLES = 1000

#: Fixed width, so a seed is named identically at any shuffle count and names sort by seed.
SHUFFLE_NAME_WIDTH = 4


def shuffle_construction_name(construction: str, seed: int) -> str:
    """Returns the `<construction>_shuffleNNNN` dataset name for one seed."""
    if seed < 1:
        raise ValueError(f"shuffle seed must be positive, got {seed}")
    if seed >= 10**SHUFFLE_NAME_WIDTH:
        raise ValueError(
            f"shuffle seed {seed} exceeds the {SHUFFLE_NAME_WIDTH}-digit dataset name width"
        )
    return f"{construction}_shuffle{seed:0{SHUFFLE_NAME_WIDTH}d}"


class HalfVerseNoded(Protocol):
    """A psalm's half-verse node ids, read-only so frozen dataclasses satisfy it."""

    @property
    def half_verse_nodes(self) -> tuple[int, ...]:
        """The BHSA node id of each half-verse, in canonical order."""


class NumberedPsalm(HalfVerseNoded, Protocol):
    """A psalm that also knows its own number, which seeds its half-verse-order permutation."""

    @property
    def number(self) -> int:
        """The psalm number, which together with the seed fixes its permutation."""


def shuffled_order_by_psalm(psalms: Sequence[NumberedPsalm], seed: int) -> dict[int, np.ndarray]:
    """One fixed-seed random permutation of indices per psalm, independent across psalms."""
    return {
        psalm.number: np.random.default_rng((psalm.number, seed)).permutation(
            len(psalm.half_verse_nodes)
        )
        for psalm in psalms
    }


def shuffled_within_half_verse_order[PsalmT: HalfVerseNoded](
    psalms: Sequence[PsalmT],
    seed: int,
    *,
    half_verses: Callable[[PsalmT], Sequence[Sequence[object]]],
) -> dict[int, np.ndarray]:
    """One fixed-seed permutation of within-half-verse element indices, keyed by (node, seed)."""
    order_by_node: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, half_verse in zip(psalm.half_verse_nodes, half_verses(psalm), strict=True):
            order_by_node[node] = np.random.default_rng((node, seed)).permutation(len(half_verse))
    return order_by_node
