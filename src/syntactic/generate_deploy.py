"""Computes and writes the psalm-scale phrase-signature deployment representation (Phase 5F)."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from core.cli import run_signature_generator
from core.export import dataset_path, write_dataset
from core.support import build_signature_vocabulary
from syntactic import DATASET_TYPE, SIGNATURE_UNIT
from syntactic.corpus import Corpus, PhrasePsalm
from syntactic.deploy import signature_deploy_vectors
from syntactic.signature_support import MIN_EXTERNAL_SUPPORT_K

CONSTRUCTION = "posmean"


def generate(
    psalms: list[PhrasePsalm], output_root: Path, external_counts: dict[str, int], k: int
) -> list[str]:
    """Writes the phrase_signature posmean dataset if not already present, returns names."""
    if dataset_path(
        output_root,
        SIGNATURE_UNIT,
        CONSTRUCTION,
        domain=DATASET_TYPE,
        unit_key="feature",
        level="phrase",
    ).exists():
        return []
    print(
        f"computing syntactic feature={SIGNATURE_UNIT} construction={CONSTRUCTION}...",
        file=sys.stderr,
    )
    vocabulary = build_signature_vocabulary(external_counts, k)
    vectors = signature_deploy_vectors(psalms, vocabulary, external_counts, k)
    description = f"Psalm-scale phrase-signature deployment [b;m], construction={CONSTRUCTION}."
    write_dataset(
        output_root,
        SIGNATURE_UNIT,
        CONSTRUCTION,
        vectors,
        description,
        domain=DATASET_TYPE,
        unit_key="feature",
        level="phrase",
    )
    return [f"{SIGNATURE_UNIT}_{CONSTRUCTION}"]


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus] = Corpus.load,
) -> None:
    """Generates the phrase_signature posmean dataset if missing."""
    run_signature_generator(
        __doc__,
        generate,
        "phrase_signature_external_support.csv",
        MIN_EXTERNAL_SUPPORT_K,
        argv,
        corpus_factory=corpus_factory,
    )


if __name__ == "__main__":
    main()
