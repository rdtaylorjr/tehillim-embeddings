"""Computes and writes the POS-only skeleton: unigram, bigram, trigram, both scales."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.cli import run_generator
from core.export import path_to_write, write_vectors
from core.parallel import map_constructions
from morphological import DATASET_TYPE
from morphological.corpus import Corpus, MorphologicalPsalm
from morphological.pos_ngram import (
    sp_1_2_3gram_psalm_vectors,
    sp_1_2_3gram_vectors,
    sp_1_2gram_psalm_vectors,
    sp_1_2gram_vectors,
    sp_unigram_psalm_vectors,
    sp_unigram_vectors,
)

_FULL_WEIGHTS = (
    "unigram",
    "1_2gram",
    "1_2_3gram",
    "unigram_psalm",
    "1_2gram_psalm",
    "1_2_3gram_psalm",
)


def _vectors_for_weight(psalms: list[MorphologicalPsalm], weight: str) -> dict[int, np.ndarray]:
    """Dispatches to the vector-building function matching `weight`."""
    if weight == "unigram":
        return sp_unigram_vectors(psalms)
    if weight == "1_2gram":
        return sp_1_2gram_vectors(psalms)
    if weight == "1_2_3gram":
        return sp_1_2_3gram_vectors(psalms)
    if weight == "unigram_psalm":
        return sp_unigram_psalm_vectors(psalms)
    if weight == "1_2gram_psalm":
        return sp_1_2gram_psalm_vectors(psalms)
    if weight == "1_2_3gram_psalm":
        return sp_1_2_3gram_psalm_vectors(psalms)
    raise ValueError(f"unknown weight {weight!r}")


@dataclass(frozen=True, slots=True)
class _Context:
    """Everything one construction needs, pickled once per worker rather than once per build."""

    psalms: tuple[MorphologicalPsalm, ...]
    output_root: Path


def write_construction(context: _Context, weight: str) -> str | None:
    """Writes one construction's dataset, or returns None when it is already written."""
    path = path_to_write(context.output_root, "sp", weight, domain=DATASET_TYPE, unit_key="feature")
    if path is None:
        return None
    vectors = _vectors_for_weight(list(context.psalms), weight)
    dimension = len(next(iter(vectors.values())))
    description = f"POS-only grammatical skeleton, construction={weight}, dimension {dimension}."
    write_vectors(path, vectors, description)
    return f"sp_{weight}"


def generate(
    psalms: list[MorphologicalPsalm], output_root: Path, *, max_workers: int | None = None
) -> list[str]:
    """Writes every not-yet-written POS construction, returns the names written."""
    context = _Context(tuple(psalms), output_root)
    return map_constructions(write_construction, context, _FULL_WEIGHTS, max_workers=max_workers)


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus] = Corpus.load,
) -> None:
    """Generates every missing POS-skeleton dataset."""
    run_generator(__doc__, generate, argv, corpus_factory=corpus_factory)


if __name__ == "__main__":
    main()
