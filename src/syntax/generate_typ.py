"""Computes and writes the phrase-type skeleton family: unigram, bigram, trigram, colon/psalm."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from lexical.export import dataset_path, write_dataset
from syntax.corpus import Corpus, PhrasePsalm
from syntax.typ_ngram import (
    phrase_typ_1_2_3gram_psalm_vectors,
    phrase_typ_1_2_3gram_vectors,
    phrase_typ_1_2gram_psalm_vectors,
    phrase_typ_1_2gram_vectors,
    phrase_typ_1gram_psalm_vectors,
    phrase_typ_1gram_vectors,
)

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
        return phrase_typ_1gram_vectors(psalms)
    if weight == "1_2gram":
        return phrase_typ_1_2gram_vectors(psalms)
    if weight == "1_2_3gram":
        return phrase_typ_1_2_3gram_vectors(psalms)
    if weight == "1gram_psalm":
        return phrase_typ_1gram_psalm_vectors(psalms)
    if weight == "1_2gram_psalm":
        return phrase_typ_1_2gram_psalm_vectors(psalms)
    if weight == "1_2_3gram_psalm":
        return phrase_typ_1_2_3gram_psalm_vectors(psalms)
    raise ValueError(f"unknown weight {weight!r}")


def generate(psalms: list[PhrasePsalm], output_root: Path) -> list[str]:
    """Writes every not-yet-written phrase-type construction, returns the names written."""
    written: list[str] = []
    for weight in _FULL_WEIGHTS:
        if dataset_path(
            output_root,
            "typ",
            weight,
            domain=_DATASET_TYPE,
            unit_key="feature",
            level="phrase",
        ).exists():
            continue
        print(f"computing syntax feature=typ construction={weight}...", file=sys.stderr)
        vectors = _vectors_for_weight(psalms, weight)
        dimension = len(next(iter(vectors.values())))
        description = f"Phrase-type-only skeleton, construction={weight}, dimension {dimension}."
        write_dataset(
            output_root,
            "typ",
            weight,
            vectors,
            description,
            domain=_DATASET_TYPE,
            unit_key="feature",
            level="phrase",
        )
        written.append(f"typ_{weight}")
    return written


def main() -> None:
    """Generates every missing phrase-type dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    output_root = parser.parse_args().output_root
    corpus = Corpus.load()
    psalms = corpus.psalms()
    written = generate(psalms, output_root)
    print(f"wrote {len(written)} dataset files", file=sys.stderr)


if __name__ == "__main__":
    main()
