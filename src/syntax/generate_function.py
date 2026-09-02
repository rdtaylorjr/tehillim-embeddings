"""Computes and writes the phrase-function skeleton: unigram, bigram, trigram, half-verse/psalm."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from core.export import dataset_path, write_sparse_vectors, write_vectors
from syntax.corpus import Corpus, PhrasePsalm
from syntax.function_ngram import (
    phrase_function_1_2_3gram_psalm_sparse_vectors,
    phrase_function_1_2_3gram_sparse_vectors,
    phrase_function_1_2gram_psalm_vectors,
    phrase_function_1_2gram_vectors,
    phrase_function_1gram_psalm_vectors,
    phrase_function_1gram_vectors,
)
from syntax.vocabulary import FUNCTION_VOCABULARY

_FULL_WEIGHTS = (
    "1gram",
    "1_2gram",
    "1_2_3gram",
    "1gram_psalm",
    "1_2gram_psalm",
    "1_2_3gram_psalm",
)

_DATASET_TYPE = "syntax"


def _vectors_for_weight(psalms: list[PhrasePsalm], weight: str) -> dict[int, np.ndarray]:
    """Dispatches to the vector-building function matching `weight`."""
    if weight == "1gram":
        return phrase_function_1gram_vectors(psalms)
    if weight == "1_2gram":
        return phrase_function_1_2gram_vectors(psalms)

    if weight == "1gram_psalm":
        return phrase_function_1gram_psalm_vectors(psalms)
    if weight == "1_2gram_psalm":
        return phrase_function_1_2gram_psalm_vectors(psalms)

    raise ValueError(f"unknown weight {weight!r}")


#: The trigram block is overwhelmingly zero at this dimension, so it is stored sparsely.
_SPARSE_WEIGHTS = {
    "1_2_3gram": phrase_function_1_2_3gram_sparse_vectors,
    "1_2_3gram_psalm": phrase_function_1_2_3gram_psalm_sparse_vectors,
}

_DIM = len(FUNCTION_VOCABULARY)
_SPARSE_DIM = _DIM + _DIM * _DIM + _DIM * _DIM * _DIM


def generate(psalms: list[PhrasePsalm], output_root: Path) -> list[str]:
    """Writes every not-yet-written phrase-function construction, returns the names written."""
    written: list[str] = []
    for weight in _FULL_WEIGHTS:
        if dataset_path(
            output_root,
            "function",
            weight,
            domain=_DATASET_TYPE,
            unit_key="feature",
            level="phrase",
        ).exists():
            continue
        print(f"computing syntax feature=function construction={weight}...", file=sys.stderr)
        path = dataset_path(
            output_root,
            "function",
            weight,
            domain=_DATASET_TYPE,
            unit_key="feature",
            level="phrase",
        )
        sparse_builder = _SPARSE_WEIGHTS.get(weight)
        if sparse_builder is not None:
            description = (
                f"Phrase-function-only skeleton, construction={weight}, dimension {_SPARSE_DIM}."
            )
            write_sparse_vectors(path, sparse_builder(psalms), _SPARSE_DIM, description)
        else:
            vectors = _vectors_for_weight(psalms, weight)
            dimension = len(next(iter(vectors.values())))
            description = (
                f"Phrase-function-only skeleton, construction={weight}, dimension {dimension}."
            )
            write_vectors(path, vectors, description)
        written.append(f"function_{weight}")
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
