"""Computes and writes the phrase-complexity structural summary: colon and psalm-broadcast."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lexical.export import dataset_path, write_dataset
from syntax.complexity import phrase_complexity_psalm_vectors, phrase_complexity_vectors
from syntax.corpus import Corpus, PhrasePsalm

_DATASET_TYPE = "syntax"
_UNIT = "complexity"


def generate(psalms: list[PhrasePsalm], output_root: Path) -> list[str]:
    """Writes both not-yet-written phrase_complexity constructions, returns the names written."""
    written: list[str] = []
    for construction, builder in (
        ("core", phrase_complexity_vectors),
        ("core_psalm", phrase_complexity_psalm_vectors),
    ):
        if dataset_path(
            output_root,
            _UNIT,
            construction,
            domain=_DATASET_TYPE,
            unit_key="feature",
            level="phrase",
        ).exists():
            continue
        print(f"computing syntax feature={_UNIT} construction={construction}...", file=sys.stderr)
        vectors = builder(psalms)
        description = (
            "Structural complexity [n_atoms; n_phrases; mean_words_per_atom; "
            f"proportion_multi_atom], construction={construction}."
        )
        write_dataset(
            output_root,
            _UNIT,
            construction,
            vectors,
            description,
            domain=_DATASET_TYPE,
            unit_key="feature",
            level="phrase",
        )
        written.append(f"{_UNIT}_{construction}")
    return written


def main() -> None:
    """Generates every missing phrase_complexity dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    output_root = parser.parse_args().output_root
    corpus = Corpus.load()
    psalms = corpus.psalms()
    written = generate(psalms, output_root)
    print(f"wrote {len(written)} dataset files", file=sys.stderr)


if __name__ == "__main__":
    main()
