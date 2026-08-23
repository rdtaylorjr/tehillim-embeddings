"""Computes and writes the phrase_signature family: inventory, 1_2gram, and 1_2_3gram."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from lexical.export import dataset_path, write_dataset
from phrase.corpus import Corpus, PhrasePsalm
from phrase.signature_support import (
    MIN_EXTERNAL_SUPPORT_K,
    build_signature_vocabulary,
    load_external_signature_counts,
)
from phrase.signature_vectorize import (
    phrase_signature_1_2_3gram_psalm_vectors,
    phrase_signature_1_2_3gram_vectors,
    phrase_signature_1_2gram_psalm_vectors,
    phrase_signature_1_2gram_vectors,
    phrase_signature_psalm_vectors,
    phrase_signature_vectors,
)

_DATASET_TYPE = "phrase"
_UNIT = "phrase_signature"


def generate(
    psalms: list[PhrasePsalm], output_root: Path, external_counts: dict[str, int], k: int
) -> list[str]:
    """Writes every not-yet-written phrase_signature construction, returns the names written."""
    vocabulary = build_signature_vocabulary(external_counts, k)

    builders: dict[str, Callable[[], dict[int, np.ndarray]]] = {
        "inventory": lambda: phrase_signature_vectors(psalms, vocabulary, external_counts, k),
        "inventory_psalm": lambda: phrase_signature_psalm_vectors(
            psalms, vocabulary, external_counts, k
        ),
        "1_2gram": lambda: phrase_signature_1_2gram_vectors(psalms, vocabulary, external_counts, k),
        "1_2gram_psalm": lambda: phrase_signature_1_2gram_psalm_vectors(
            psalms, vocabulary, external_counts, k
        ),
        "1_2_3gram": lambda: phrase_signature_1_2_3gram_vectors(
            psalms, vocabulary, external_counts, k
        ),
        "1_2_3gram_psalm": lambda: phrase_signature_1_2_3gram_psalm_vectors(
            psalms, vocabulary, external_counts, k
        ),
    }

    written: list[str] = []
    for construction, builder in builders.items():
        if dataset_path(output_root, _UNIT, construction, dataset_type=_DATASET_TYPE).exists():
            continue
        print(f"computing phrase unit={_UNIT} construction={construction}...", file=sys.stderr)
        description = (
            f"Phrase-signature histogram (RARE-collapsed, k={k}), construction={construction}."
        )
        write_dataset(
            output_root, _UNIT, construction, builder(), description, dataset_type=_DATASET_TYPE
        )
        written.append(f"{_UNIT}_{construction}")
    return written


def main() -> None:
    """Generates every missing phrase_signature dataset."""
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
