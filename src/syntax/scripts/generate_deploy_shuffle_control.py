"""Generates N colon-order-shuffled phrase_signature_posmean datasets, a shuffle-null control."""

from __future__ import annotations

import sys
from pathlib import Path

from lexical.export import write_dataset
from lexical.shuffle_control import shuffled_order_by_psalm
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


def generate_shuffle_control(
    psalms: list[PhrasePsalm],
    output_root: Path,
    n_shuffles: int,
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
) -> list[str]:
    """Writes n_shuffles seeded, colon-order-shuffled phrase_signature_posmean datasets."""
    written: list[str] = []
    for seed in range(1, n_shuffles + 1):
        order = shuffled_order_by_psalm(psalms, seed)  # type: ignore[arg-type]
        vectors = signature_deploy_vectors(
            psalms, vocabulary, external_counts, k, order_by_psalm=order
        )
        weight = f"{_CONSTRUCTION}_shuffle{seed:02d}"
        description = f"Shuffle-null order-effect control for {_UNIT}_{_CONSTRUCTION}, seed {seed}."
        write_dataset(
            output_root,
            _UNIT,
            weight,
            vectors,
            description,
            domain=_DATASET_TYPE,
            unit_key="feature",
            level="phrase",
        )
        written.append(f"{_UNIT}_{weight}")
    return written


def main() -> None:
    """Generates the shuffle-null control datasets for phrase_signature_posmean."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--n-shuffles", type=int, default=30)
    args = parser.parse_args()

    corpus = Corpus.load()
    psalms = corpus.psalms()
    support_path = args.config_root / "phrase_signature_external_support.csv"
    external_counts = load_external_signature_counts(support_path)
    vocabulary = build_signature_vocabulary(external_counts, MIN_EXTERNAL_SUPPORT_K)
    written = generate_shuffle_control(
        psalms,
        args.output_root,
        args.n_shuffles,
        vocabulary,
        external_counts,
        MIN_EXTERNAL_SUPPORT_K,
    )
    print(f"wrote {len(written)} shuffle-control datasets", file=sys.stderr)


if __name__ == "__main__":
    main()
