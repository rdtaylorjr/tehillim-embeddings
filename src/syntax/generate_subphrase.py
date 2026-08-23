"""Computes and writes the safe subphrase-relation skeleton: `rela=par` masked, colon and psalm."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lexical.export import dataset_path, write_dataset
from syntax.corpus import Corpus, PhrasePsalm
from syntax.subphrase_vectorize import (
    subphrase_rela_1gram_psalm_vectors,
    subphrase_rela_1gram_vectors,
)

_DATASET_TYPE = "syntax"
_UNIT = "subphrase_rela"


def generate(psalms: list[PhrasePsalm], output_root: Path) -> list[str]:
    """Writes both not-yet-written phrase_subphrase_rela constructions, returns the names."""
    written: list[str] = []
    for construction, builder in (
        ("1gram", subphrase_rela_1gram_vectors),
        ("1gram_psalm", subphrase_rela_1gram_psalm_vectors),
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
        description = f"Safe subphrase-relation skeleton (par masked), construction={construction}."
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
    """Generates every missing phrase_subphrase_rela dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    output_root = parser.parse_args().output_root
    corpus = Corpus.load()
    psalms = corpus.psalms()
    written = generate(psalms, output_root)
    print(f"wrote {len(written)} dataset files", file=sys.stderr)


if __name__ == "__main__":
    main()
