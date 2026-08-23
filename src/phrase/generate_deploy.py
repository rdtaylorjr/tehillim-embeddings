"""Computes and writes the psalm-scale phrase-signature deployment representation (Phase 5F)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lexical.export import dataset_path, write_dataset
from phrase.corpus import Corpus, PhrasePsalm
from phrase.deploy import signature_deploy_vectors
from phrase.signature_support import (
    MIN_EXTERNAL_SUPPORT_K,
    build_signature_vocabulary,
    load_external_signature_counts,
)

_DATASET_TYPE = "phrase"
_UNIT = "phrase_signature"
_CONSTRUCTION = "posmean"


def generate(
    psalms: list[PhrasePsalm], output_root: Path, external_counts: dict[str, int], k: int
) -> list[str]:
    """Writes the phrase_signature posmean dataset if not already present, returns names."""
    if dataset_path(output_root, _UNIT, _CONSTRUCTION, dataset_type=_DATASET_TYPE).exists():
        return []
    print(f"computing phrase unit={_UNIT} construction={_CONSTRUCTION}...", file=sys.stderr)
    vocabulary = build_signature_vocabulary(external_counts, k)
    vectors = signature_deploy_vectors(psalms, vocabulary, external_counts, k)
    description = f"Psalm-scale phrase-signature deployment [b;m], construction={_CONSTRUCTION}."
    write_dataset(
        output_root, _UNIT, _CONSTRUCTION, vectors, description, dataset_type=_DATASET_TYPE
    )
    return [f"{_UNIT}_{_CONSTRUCTION}"]


def main() -> None:
    """Generates the phrase_signature posmean dataset if missing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    output_root = parser.parse_args().output_root
    corpus = Corpus.load()
    psalms = corpus.psalms()
    support_path = output_root / "config" / "phrase_signature_external_support.csv"
    external_counts = load_external_signature_counts(support_path)
    written = generate(psalms, output_root, external_counts, MIN_EXTERNAL_SUPPORT_K)
    print(f"wrote {len(written)} dataset files", file=sys.stderr)


if __name__ == "__main__":
    main()
