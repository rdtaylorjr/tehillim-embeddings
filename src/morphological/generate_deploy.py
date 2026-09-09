"""Computes and writes the psalm-scale grammatical deployment representation (Phase 4E)."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from core.cli import run_generator
from core.export import dataset_path, write_dataset
from morphological import DATASET_TYPE, SUFFIX_UNIT
from morphological.corpus import Corpus, MorphologicalPsalm
from morphological.deploy import suffix_deploy_vectors

CONSTRUCTION = "posmean"


def generate(psalms: list[MorphologicalPsalm], output_root: Path) -> list[str]:
    """Writes the morph_suffix posmean dataset if not already present, returns the names written."""
    if dataset_path(
        output_root, SUFFIX_UNIT, CONSTRUCTION, domain=DATASET_TYPE, unit_key="feature"
    ).exists():
        return []
    print(
        f"computing morphological feature={SUFFIX_UNIT} construction={CONSTRUCTION}...",
        file=sys.stderr,
    )
    vectors = suffix_deploy_vectors(psalms)
    description = (
        f"Psalm-scale grammatical deployment [b;m] over the suffix vocabulary, "
        f"construction={CONSTRUCTION}."
    )
    write_dataset(
        output_root,
        SUFFIX_UNIT,
        CONSTRUCTION,
        vectors,
        description,
        domain=DATASET_TYPE,
        unit_key="feature",
    )
    return [f"{SUFFIX_UNIT}_{CONSTRUCTION}"]


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus] = Corpus.load,
) -> None:
    """Generates the morph_suffix posmean dataset if missing."""
    run_generator(__doc__, generate, argv, corpus_factory=corpus_factory)


if __name__ == "__main__":
    main()
