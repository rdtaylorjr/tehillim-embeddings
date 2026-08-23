"""Computes and writes the safe phrase-relation skeleton: `rela=Para` masked, colon and psalm."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lexical.export import dataset_path, write_dataset
from phrase.corpus import Corpus, PhrasePsalm
from phrase.rela_vectorize import phrase_rela_1gram_psalm_vectors, phrase_rela_1gram_vectors

_DATASET_TYPE = "phrase"
_UNIT = "phrase_rela"


def generate(psalms: list[PhrasePsalm], output_root: Path) -> list[str]:
    """Writes both not-yet-written phrase_rela constructions, returns the names written."""
    written: list[str] = []
    for construction, builder in (
        ("1gram", phrase_rela_1gram_vectors),
        ("1gram_psalm", phrase_rela_1gram_psalm_vectors),
    ):
        if dataset_path(output_root, _UNIT, construction, dataset_type=_DATASET_TYPE).exists():
            continue
        print(f"computing phrase unit={_UNIT} construction={construction}...", file=sys.stderr)
        vectors = builder(psalms)
        description = f"Safe phrase-relation skeleton (Para masked), construction={construction}."
        write_dataset(
            output_root, _UNIT, construction, vectors, description, dataset_type=_DATASET_TYPE
        )
        written.append(f"{_UNIT}_{construction}")
    return written


def main() -> None:
    """Generates every missing phrase_rela dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    output_root = parser.parse_args().output_root
    corpus = Corpus.load()
    psalms = corpus.psalms()
    written = generate(psalms, output_root)
    print(f"wrote {len(written)} dataset files", file=sys.stderr)


if __name__ == "__main__":
    main()
