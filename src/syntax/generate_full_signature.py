"""Computes and writes the full (typ:function:det) signature inventory: H5.8's S+det test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lexical.export import dataset_path, write_dataset
from syntax.corpus import Corpus, PhrasePsalm
from syntax.full_signature_vectorize import (
    phrase_full_signature_psalm_vectors,
    phrase_full_signature_vectors,
)
from syntax.signature_support import (
    MIN_EXTERNAL_SUPPORT_K_FULL,
    build_signature_vocabulary,
    load_external_signature_counts,
)

_DATASET_TYPE = "syntax"
_UNIT = "full_signature"


def generate(
    psalms: list[PhrasePsalm], output_root: Path, external_counts: dict[str, int], k: int
) -> list[str]:
    """Writes both not-yet-written phrase_full_signature constructions, returns the names."""
    vocabulary = build_signature_vocabulary(external_counts, k)

    written: list[str] = []
    for construction, builder in (
        ("inventory", phrase_full_signature_vectors),
        ("inventory_psalm", phrase_full_signature_psalm_vectors),
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
        vectors = builder(psalms, vocabulary, external_counts, k)
        description = (
            f"Full typ:function:det signature histogram (RARE-collapsed, k={k}), "
            f"construction={construction}."
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
    """Generates every missing phrase_full_signature dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root
    config_root = args.config_root
    corpus = Corpus.load()
    psalms = corpus.psalms()
    support_path = config_root / "phrase_full_signature_external_support.csv"
    external_counts = load_external_signature_counts(support_path)
    written = generate(psalms, output_root, external_counts, MIN_EXTERNAL_SUPPORT_K_FULL)
    print(f"wrote {len(written)} dataset files", file=sys.stderr)


if __name__ == "__main__":
    main()
