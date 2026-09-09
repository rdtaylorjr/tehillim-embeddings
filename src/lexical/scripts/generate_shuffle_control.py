"""Generates N order-shuffled icf_position_mean_psalm datasets, a shuffle-null control."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from core.columns import PsalmColumns
from lexical.corpus import Corpus
from lexical.psalm_zoning import psalm_position_mean_vectors
from lexical.scripts import shuffle_driver
from lexical.scripts.shuffle_driver import build_parser, generate

_CONSTRUCTION = "icf_position_mean_psalm"

__all__ = ["build_parser", "generate", "main"]


def build_vectors(
    columns: list[PsalmColumns],
    vocabulary: tuple[str, ...],
    icf_weights: dict[str, float],
    order: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """One seed's icf_position_mean_psalm vectors under a shuffled half-verse order."""
    return psalm_position_mean_vectors(columns, vocabulary, icf_weights, order_by_psalm=order)


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus] = Corpus.load,
) -> None:
    """Generates the shuffle-null control datasets for icf_position_mean_psalm."""
    shuffle_driver.run(__doc__, _CONSTRUCTION, build_vectors, argv, corpus_factory=corpus_factory)


if __name__ == "__main__":
    main()
