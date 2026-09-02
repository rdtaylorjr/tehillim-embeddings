"""Generates N order-shuffled icf_position4 datasets, a shuffle-null control (parallelism)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from core.columns import PsalmColumns
from lexical.corpus import Corpus
from lexical.positional import positional_icf_vectors
from lexical.scripts import shuffle_driver
from lexical.scripts.shuffle_driver import build_parser, generate

_K = 4
_CONSTRUCTION = "icf_position4"

__all__ = ["build_parser", "generate", "main"]


def build_vectors(
    columns: list[PsalmColumns],
    vocabulary: tuple[str, ...],
    icf_weights: dict[str, float],
    order: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """One seed's icf_position4 vectors under a shuffled half-verse order."""
    return positional_icf_vectors(columns, vocabulary, icf_weights, k=_K, order_by_psalm=order)


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus] = Corpus.load,
) -> None:
    """Generates the shuffle-null control datasets for icf_position4."""
    shuffle_driver.run(__doc__, _CONSTRUCTION, build_vectors, argv, corpus_factory=corpus_factory)


if __name__ == "__main__":
    main()
