"""Computes and writes the psalm-scale phrase-signature deployment representation (Phase 5F)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lexical.export import dataset_path, write_dataset
from syntax.corpus import Corpus, PhrasePsalm
from syntax.deploy import signature_deploy_vectors
from syntax.signature_support import (
    MIN_EXTERNAL_SUPPORT_K,
    build_signature_vocabulary,
    load_external_signature_counts,
)

_DATASET_TYPE = "syntax"
_UNIT = "signature"
_CONSTRUCTION = "posmean"


def generate(
    psalms: list[PhrasePsalm], output_root: Path, external_counts: dict[str, int], k: int
) -> list[str]:
    """Writes the phrase_signature posmean dataset if not already present, returns names."""
    if dataset_path(
        output_root,
        _UNIT,
        _CONSTRUCTION,
        domain=_DATASET_TYPE,
        unit_key="feature",
        level="phrase",
    ).exists():
        return []
    print(f"computing syntax feature={_UNIT} construction={_CONSTRUCTION}...", file=sys.stderr)
    vocabulary = build_signature_vocabulary(external_counts, k)
    vectors = signature_deploy_vectors(psalms, vocabulary, external_counts, k)
    description = f"Psalm-scale phrase-signature deployment [b;m], construction={_CONSTRUCTION}."
    write_dataset(
        output_root,
        _UNIT,
        _CONSTRUCTION,
        vectors,
        description,
        domain=_DATASET_TYPE,
        unit_key="feature",
        level="phrase",
    )
    return [f"{_UNIT}_{_CONSTRUCTION}"]


def main() -> None:
    """Generates the phrase_signature posmean dataset if missing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root
    config_root = args.config_root
    corpus = Corpus.load()
    psalms = corpus.psalms()
    support_path = config_root / "phrase_signature_external_support.csv"
    external_counts = load_external_signature_counts(support_path)
    written = generate(psalms, output_root, external_counts, MIN_EXTERNAL_SUPPORT_K)
    print(f"wrote {len(written)} dataset files", file=sys.stderr)


if __name__ == "__main__":
    main()
