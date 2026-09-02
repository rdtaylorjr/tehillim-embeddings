"""Generates N half-verse-order-shuffled morph_suffix_posmean datasets, a shuffle-null control."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core.export import write_dataset
from core.parallel import map_seeds
from core.shuffle import DEFAULT_N_SHUFFLES, shuffle_construction_name, shuffled_order_by_psalm
from morphology.corpus import Corpus, MorphologicalPsalm
from morphology.deploy import suffix_deploy_vectors

_DATASET_TYPE = "morphology"
_UNIT = "morph_suffix"
_CONSTRUCTION = "posmean"


@dataclass(frozen=True, slots=True)
class _SeedContext:
    """Everything one seed needs, pickled once per worker process rather than once per seed."""

    psalms: tuple[MorphologicalPsalm, ...]
    output_root: Path


def write_seed(context: _SeedContext, seed: int) -> str:
    """Writes one seeded, half-verse-order-shuffled dataset and returns its qualified name."""
    psalms = list(context.psalms)
    order = shuffled_order_by_psalm(psalms, seed)
    vectors = suffix_deploy_vectors(psalms, order_by_psalm=order)
    name = shuffle_construction_name(_CONSTRUCTION, seed)
    description = f"Shuffle-null order-effect control for {_UNIT}_{_CONSTRUCTION}, seed {seed}."
    write_dataset(
        context.output_root,
        _UNIT,
        name,
        vectors,
        description,
        domain=_DATASET_TYPE,
        unit_key="feature",
    )
    return f"{_UNIT}_{name}"


def generate_shuffle_control(
    psalms: list[MorphologicalPsalm],
    output_root: Path,
    n_shuffles: int,
    *,
    max_workers: int | None = None,
) -> list[str]:
    """Writes n_shuffles seeded, half-verse-order-shuffled morph_suffix_posmean datasets."""
    context = _SeedContext(psalms=tuple(psalms), output_root=output_root)
    return map_seeds(write_seed, context, range(1, n_shuffles + 1), max_workers=max_workers)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for the morph_suffix_posmean shuffle-null control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES)
    parser.add_argument("--max-workers", type=int, default=None)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus] = Corpus.load,
) -> None:
    """Generates the shuffle-null control datasets for morph_suffix_posmean."""
    args = build_parser().parse_args(argv)

    corpus = corpus_factory()
    psalms = corpus.psalms()
    written = generate_shuffle_control(
        psalms, args.output_root, args.n_shuffles, max_workers=args.max_workers
    )
    print(f"wrote {len(written)} shuffle-control datasets", file=sys.stderr)


if __name__ == "__main__":
    main()
