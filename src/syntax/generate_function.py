"""Computes and writes the phrase-function skeleton: unigram, bigram, trigram, half-verse/psalm."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.cli import run_generator
from core.export import path_to_write, write_sparse_vectors, write_vectors
from core.parallel import map_constructions
from syntax import DATASET_TYPE
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


@dataclass(frozen=True, slots=True)
class _Context:
    """Everything one construction needs, pickled once per worker rather than once per build."""

    psalms: tuple[PhrasePsalm, ...]
    output_root: Path


def write_construction(context: _Context, weight: str) -> str | None:
    """Writes one construction's dataset, or returns None when it is already written."""
    path = path_to_write(
        context.output_root,
        "function",
        weight,
        domain=DATASET_TYPE,
        unit_key="feature",
        level="phrase",
    )
    if path is None:
        return None
    psalms = list(context.psalms)
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
    return f"function_{weight}"


def generate(
    psalms: list[PhrasePsalm], output_root: Path, *, max_workers: int | None = None
) -> list[str]:
    """Writes every not-yet-written phrase-function construction, returns the names written."""
    context = _Context(tuple(psalms), output_root)
    return map_constructions(write_construction, context, _FULL_WEIGHTS, max_workers=max_workers)


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus] = Corpus.load,
) -> None:
    """Generates every missing phrase-function dataset."""
    run_generator(__doc__, generate, argv, corpus_factory=corpus_factory)


if __name__ == "__main__":
    main()
