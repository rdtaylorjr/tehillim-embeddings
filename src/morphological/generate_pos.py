"""Computes and writes the POS-only skeleton family: unigram, bigram, trigram, colon and psalm."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from lexical.export import dataset_path, write_dataset
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

_DATASET_TYPE = "morphological"


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


def generate(psalms: list[MorphologicalPsalm], output_root: Path) -> list[str]:
    """Writes every not-yet-written POS construction, returns the names written."""
    written: list[str] = []
    for weight in _FULL_WEIGHTS:
        if dataset_path(output_root, "sp", weight, dataset_type=_DATASET_TYPE).exists():
            continue
        print(f"computing morphological unit=sp construction={weight}...", file=sys.stderr)
        vectors = _vectors_for_weight(psalms, weight)
        dimension = len(next(iter(vectors.values())))
        description = (
            f"POS-only grammatical skeleton, construction={weight}, dimension {dimension}."
        )
        write_dataset(output_root, "sp", weight, vectors, description, dataset_type=_DATASET_TYPE)
        written.append(f"sp_{weight}")
    return written


def main() -> None:
    """Generates every missing POS-skeleton dataset."""
    output_root = Path(__file__).resolve().parents[2]
    corpus = Corpus.load()
    psalms = corpus.psalms()
    written = generate(psalms, output_root)
    print(f"wrote {len(written)} dataset files", file=sys.stderr)


if __name__ == "__main__":
    main()
