"""Computes and writes the phrase-function skeleton: unigram, bigram, trigram, colon/psalm."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from lexical.export import dataset_path, write_dataset
from phrase.corpus import Corpus, PhrasePsalm
from phrase.function_ngram import (
    phrase_function_1_2_3gram_psalm_vectors,
    phrase_function_1_2_3gram_vectors,
    phrase_function_1_2gram_psalm_vectors,
    phrase_function_1_2gram_vectors,
    phrase_function_1gram_psalm_vectors,
    phrase_function_1gram_vectors,
)

_FULL_WEIGHTS = (
    "1gram",
    "1_2gram",
    "1_2_3gram",
    "1gram_psalm",
    "1_2gram_psalm",
    "1_2_3gram_psalm",
)

_DATASET_TYPE = "phrase"


def _vectors_for_weight(psalms: list[PhrasePsalm], weight: str) -> dict[int, np.ndarray]:
    """Dispatches to the vector-building function matching `weight`."""
    if weight == "1gram":
        return phrase_function_1gram_vectors(psalms)
    if weight == "1_2gram":
        return phrase_function_1_2gram_vectors(psalms)
    if weight == "1_2_3gram":
        return phrase_function_1_2_3gram_vectors(psalms)
    if weight == "1gram_psalm":
        return phrase_function_1gram_psalm_vectors(psalms)
    if weight == "1_2gram_psalm":
        return phrase_function_1_2gram_psalm_vectors(psalms)
    if weight == "1_2_3gram_psalm":
        return phrase_function_1_2_3gram_psalm_vectors(psalms)
    raise ValueError(f"unknown weight {weight!r}")


def generate(psalms: list[PhrasePsalm], output_root: Path) -> list[str]:
    """Writes every not-yet-written phrase-function construction, returns the names written."""
    written: list[str] = []
    for weight in _FULL_WEIGHTS:
        if dataset_path(
            output_root, "phrase_function", weight, dataset_type=_DATASET_TYPE
        ).exists():
            continue
        print(f"computing phrase unit=phrase_function construction={weight}...", file=sys.stderr)
        vectors = _vectors_for_weight(psalms, weight)
        dimension = len(next(iter(vectors.values())))
        description = (
            f"Phrase-function-only skeleton, construction={weight}, dimension {dimension}."
        )
        write_dataset(
            output_root,
            "phrase_function",
            weight,
            vectors,
            description,
            dataset_type=_DATASET_TYPE,
        )
        written.append(f"phrase_function_{weight}")
    return written


def main() -> None:
    """Generates every missing phrase-function dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    output_root = parser.parse_args().output_root
    corpus = Corpus.load()
    psalms = corpus.psalms()
    written = generate(psalms, output_root)
    print(f"wrote {len(written)} dataset files", file=sys.stderr)


if __name__ == "__main__":
    main()
