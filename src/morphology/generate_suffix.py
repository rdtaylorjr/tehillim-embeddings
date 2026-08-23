"""Computes and writes the morph_suffix family: suffix inventory, and host-signature-plus-suffix."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from lexical.export import dataset_path, write_dataset
from morphology.corpus import Corpus, MorphologicalPsalm
from morphology.signature_support import (
    MIN_EXTERNAL_SUPPORT_K,
    build_signature_vocabulary,
    load_external_signature_counts,
)
from morphology.suffix import (
    host_plus_suffix_psalm_vectors,
    host_plus_suffix_vectors,
    suffix_inventory_psalm_vectors,
    suffix_inventory_vectors,
)

_DATASET_TYPE = "morphology"


def generate(
    psalms: list[MorphologicalPsalm],
    output_root: Path,
    external_counts: dict[str, int],
    k: int,
) -> list[str]:
    """Writes every not-yet-written morph_suffix construction, returns the names written."""
    signature_vocabulary = build_signature_vocabulary(external_counts, k)

    builders: dict[str, Callable[[], dict[int, np.ndarray]]] = {
        "inventory": lambda: suffix_inventory_vectors(psalms),
        "inventory_psalm": lambda: suffix_inventory_psalm_vectors(psalms),
        "host_plus_suffix": lambda: host_plus_suffix_vectors(
            psalms, signature_vocabulary, external_counts, k
        ),
        "host_plus_suffix_psalm": lambda: host_plus_suffix_psalm_vectors(
            psalms, signature_vocabulary, external_counts, k
        ),
    }

    written: list[str] = []
    for construction, builder in builders.items():
        if dataset_path(
            output_root,
            "morph_suffix",
            construction,
            domain=_DATASET_TYPE,
            unit_key="feature",
        ).exists():
            continue
        print(
            f"computing morphology feature=morph_suffix construction={construction}...",
            file=sys.stderr,
        )
        description = f"Pronominal-suffix representation, construction={construction}."
        write_dataset(
            output_root,
            "morph_suffix",
            construction,
            builder(),
            description,
            domain=_DATASET_TYPE,
            unit_key="feature",
        )
        written.append(f"morph_suffix_{construction}")

    return written


def main() -> None:
    """Generates every missing morph_suffix dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root
    config_root = args.config_root
    corpus = Corpus.load()
    psalms = corpus.psalms()
    support_path = config_root / "morph_signature_external_support.csv"
    external_counts = load_external_signature_counts(support_path)
    written = generate(psalms, output_root, external_counts, MIN_EXTERNAL_SUPPORT_K)
    print(f"wrote {len(written)} dataset files", file=sys.stderr)


if __name__ == "__main__":
    main()
