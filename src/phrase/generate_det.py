"""Computes and writes the phrase-determination (det) skeleton: colon and psalm-broadcast."""

from __future__ import annotations

import sys
from pathlib import Path

from lexical.export import dataset_path, write_dataset
from phrase.corpus import Corpus, PhrasePsalm
from phrase.det_vectorize import phrase_det_1gram_psalm_vectors, phrase_det_1gram_vectors

_DATASET_TYPE = "phrase"
_UNIT = "phrase_det"


def generate(psalms: list[PhrasePsalm], output_root: Path) -> list[str]:
    """Writes both not-yet-written phrase_det constructions, returns the names written."""
    written: list[str] = []
    for construction, builder in (
        ("1gram", phrase_det_1gram_vectors),
        ("1gram_psalm", phrase_det_1gram_psalm_vectors),
    ):
        if dataset_path(output_root, _UNIT, construction, dataset_type=_DATASET_TYPE).exists():
            continue
        print(f"computing phrase unit={_UNIT} construction={construction}...", file=sys.stderr)
        vectors = builder(psalms)
        description = f"Phrase-determination skeleton, construction={construction}."
        write_dataset(
            output_root, _UNIT, construction, vectors, description, dataset_type=_DATASET_TYPE
        )
        written.append(f"{_UNIT}_{construction}")
    return written


def main() -> None:
    """Generates every missing phrase_det dataset."""
    output_root = Path(__file__).resolve().parents[2]
    corpus = Corpus.load()
    psalms = corpus.psalms()
    written = generate(psalms, output_root)
    print(f"wrote {len(written)} dataset files", file=sys.stderr)


if __name__ == "__main__":
    main()
