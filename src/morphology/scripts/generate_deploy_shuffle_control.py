"""Generates N colon-order-shuffled morph_suffix_posmean datasets, a shuffle-null control."""

from __future__ import annotations

import sys
from pathlib import Path

from lexical.export import write_dataset
from lexical.shuffle_control import shuffled_order_by_psalm
from morphology.corpus import Corpus, MorphologicalPsalm
from morphology.deploy import suffix_deploy_vectors

_DATASET_TYPE = "morphology"
_UNIT = "morph_suffix"
_CONSTRUCTION = "posmean"


def generate_shuffle_control(
    psalms: list[MorphologicalPsalm], output_root: Path, n_shuffles: int
) -> list[str]:
    """Writes n_shuffles seeded, colon-order-shuffled morph_suffix_posmean datasets."""
    written: list[str] = []
    for seed in range(1, n_shuffles + 1):
        order = shuffled_order_by_psalm(psalms, seed)  # type: ignore[arg-type]
        vectors = suffix_deploy_vectors(psalms, order_by_psalm=order)
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
        )
        written.append(f"{_UNIT}_{weight}")
    return written


def main() -> None:
    """Generates the shuffle-null control datasets for morph_suffix_posmean."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--n-shuffles", type=int, default=30)
    args = parser.parse_args()

    corpus = Corpus.load()
    psalms = corpus.psalms()
    written = generate_shuffle_control(psalms, args.output_root, args.n_shuffles)
    print(f"wrote {len(written)} shuffle-control datasets", file=sys.stderr)


if __name__ == "__main__":
    main()
