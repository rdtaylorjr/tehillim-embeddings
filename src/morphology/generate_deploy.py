"""Computes and writes the psalm-scale grammatical deployment representation (Phase 4E)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lexical.export import dataset_path, write_dataset
from morphology.corpus import Corpus, MorphologicalPsalm
from morphology.deploy import suffix_deploy_vectors

_DATASET_TYPE = "morphology"
_UNIT = "morph_suffix"
_CONSTRUCTION = "posmean"


def generate(psalms: list[MorphologicalPsalm], output_root: Path) -> list[str]:
    """Writes the morph_suffix posmean dataset if not already present, returns the names written."""
    if dataset_path(
        output_root, _UNIT, _CONSTRUCTION, domain=_DATASET_TYPE, unit_key="feature"
    ).exists():
        return []
    print(f"computing morphology feature={_UNIT} construction={_CONSTRUCTION}...", file=sys.stderr)
    vectors = suffix_deploy_vectors(psalms)
    description = (
        f"Psalm-scale grammatical deployment [b;m] over the suffix vocabulary, "
        f"construction={_CONSTRUCTION}."
    )
    write_dataset(
        output_root,
        _UNIT,
        _CONSTRUCTION,
        vectors,
        description,
        domain=_DATASET_TYPE,
        unit_key="feature",
    )
    return [f"{_UNIT}_{_CONSTRUCTION}"]


def main() -> None:
    """Generates the morph_suffix posmean dataset if missing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    output_root = parser.parse_args().output_root
    corpus = Corpus.load()
    psalms = corpus.psalms()
    written = generate(psalms, output_root)
    print(f"wrote {len(written)} dataset files", file=sys.stderr)


if __name__ == "__main__":
    main()
