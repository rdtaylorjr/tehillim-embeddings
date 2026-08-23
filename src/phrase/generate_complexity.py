"""Computes and writes the phrase-complexity structural summary: colon and psalm-broadcast."""

from __future__ import annotations

import sys
from pathlib import Path

from lexical.export import dataset_path, write_dataset
from phrase.complexity import phrase_complexity_psalm_vectors, phrase_complexity_vectors
from phrase.corpus import Corpus, PhrasePsalm

_DATASET_TYPE = "phrase"
_UNIT = "phrase_complexity"


def generate(psalms: list[PhrasePsalm], output_root: Path) -> list[str]:
    """Writes both not-yet-written phrase_complexity constructions, returns the names written."""
    written: list[str] = []
    for construction, builder in (
        ("core", phrase_complexity_vectors),
        ("core_psalm", phrase_complexity_psalm_vectors),
    ):
        if dataset_path(output_root, _UNIT, construction, dataset_type=_DATASET_TYPE).exists():
            continue
        print(f"computing phrase unit={_UNIT} construction={construction}...", file=sys.stderr)
        vectors = builder(psalms)
        description = (
            "Structural complexity [n_atoms; n_phrases; mean_words_per_atom; "
            f"proportion_multi_atom], construction={construction}."
        )
        write_dataset(
            output_root, _UNIT, construction, vectors, description, dataset_type=_DATASET_TYPE
        )
        written.append(f"{_UNIT}_{construction}")
    return written


def main() -> None:
    """Generates every missing phrase_complexity dataset."""
    output_root = Path(__file__).resolve().parents[2]
    corpus = Corpus.load()
    psalms = corpus.psalms()
    written = generate(psalms, output_root)
    print(f"wrote {len(written)} dataset files", file=sys.stderr)


if __name__ == "__main__":
    main()
