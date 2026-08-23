"""Computes and writes the `[phrase_typ; phrase_function]` marginal baseline, colon and psalm."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lexical.export import dataset_path, write_dataset
from phrase.corpus import Corpus, PhrasePsalm
from phrase.marginal import typ_function_marginal_psalm_vectors, typ_function_marginal_vectors

_DATASET_TYPE = "phrase"
_UNIT = "phrase_marginal"


def generate(psalms: list[PhrasePsalm], output_root: Path) -> list[str]:
    """Writes both not-yet-written phrase_marginal constructions, returns the names written."""
    written: list[str] = []
    for construction, builder in (
        ("typ_function", typ_function_marginal_vectors),
        ("typ_function_psalm", typ_function_marginal_psalm_vectors),
    ):
        if dataset_path(output_root, _UNIT, construction, dataset_type=_DATASET_TYPE).exists():
            continue
        print(f"computing phrase unit={_UNIT} construction={construction}...", file=sys.stderr)
        vectors = builder(psalms)
        description = (
            f"Independent [typ; function] marginal histograms, construction={construction}."
        )
        write_dataset(
            output_root, _UNIT, construction, vectors, description, dataset_type=_DATASET_TYPE
        )
        written.append(f"{_UNIT}_{construction}")
    return written


def main() -> None:
    """Generates every missing phrase_marginal dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    output_root = parser.parse_args().output_root
    corpus = Corpus.load()
    psalms = corpus.psalms()
    written = generate(psalms, output_root)
    print(f"wrote {len(written)} dataset files", file=sys.stderr)


if __name__ == "__main__":
    main()
