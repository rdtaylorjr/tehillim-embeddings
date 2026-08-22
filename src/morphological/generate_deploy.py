"""Computes and writes the psalm-scale grammatical deployment representation (Phase 4E)."""

from __future__ import annotations

import sys
from pathlib import Path

from lexical.export import dataset_path, write_dataset
from morphological.corpus import Corpus, MorphologicalPsalm
from morphological.deploy import suffix_deploy_vectors

_DATASET_TYPE = "morphological"
_UNIT = "morph_suffix"
_CONSTRUCTION = "posmean"


def generate(psalms: list[MorphologicalPsalm], output_root: Path) -> list[str]:
    """Writes the morph_suffix posmean dataset if not already present, returns the names written."""
    if dataset_path(output_root, _UNIT, _CONSTRUCTION, dataset_type=_DATASET_TYPE).exists():
        return []
    print(f"computing morphological unit={_UNIT} construction={_CONSTRUCTION}...", file=sys.stderr)
    vectors = suffix_deploy_vectors(psalms)
    description = (
        f"Psalm-scale grammatical deployment [b;m] over the suffix vocabulary, "
        f"construction={_CONSTRUCTION}."
    )
    write_dataset(
        output_root, _UNIT, _CONSTRUCTION, vectors, description, dataset_type=_DATASET_TYPE
    )
    return [f"{_UNIT}_{_CONSTRUCTION}"]


def main() -> None:
    """Generates the morph_suffix posmean dataset if missing."""
    output_root = Path(__file__).resolve().parents[2]
    corpus = Corpus.load()
    psalms = corpus.psalms()
    written = generate(psalms, output_root)
    print(f"wrote {len(written)} dataset files", file=sys.stderr)


if __name__ == "__main__":
    main()
