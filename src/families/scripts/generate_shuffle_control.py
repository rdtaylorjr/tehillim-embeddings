"""Writes N seeded draws of one order-sensitive family: the shuffle-null control its key names."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from core.cli import (
    add_config_root_argument,
    add_output_root_argument,
    add_shuffle_arguments,
    report_written,
)
from core.export import write_sparse_vectors, write_vectors
from core.parallel import map_seeds
from families.shuffle import FAMILIES, Draws, dataset_target, draw, load_draws

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class SeedContext:
    """Everything one seed needs, pickled once per worker process rather than once per seed."""

    draws: Draws
    output_root: Path


def write_seed(context: SeedContext, seed: int) -> str:
    """Writes one seeded draw and returns the construction name it was written under."""
    draws = context.draws
    path = dataset_target(draws.key, context.output_root, seed)
    description = f"Shuffle-null order-effect control for {draws.key}, seed {seed}."
    vectors = draw(draws, seed)
    if draws.sparse_width is None:
        write_vectors(path, vectors, description)
    else:
        write_sparse_vectors(path, vectors, draws.sparse_width, description)
    return path.parent.name


def generate(
    draws: Draws, output_root: Path, n_shuffles: int, *, max_workers: int | None = None
) -> list[str]:
    """Writes one dataset per seed across worker processes, returning the names written."""
    context = SeedContext(draws=draws, output_root=output_root)
    return map_seeds(write_seed, context, range(1, n_shuffles + 1), max_workers=max_workers)


def build_parser() -> argparse.ArgumentParser:
    """The command line every family shares, since one generator now writes all of them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=sorted(FAMILIES), metavar="KEY")
    add_output_root_argument(parser)
    add_config_root_argument(parser)
    add_shuffle_arguments(parser)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    draws_factory: Callable[[str, Path], Draws] = load_draws,
) -> None:
    """Generates the shuffle-null control datasets for one registered family."""
    args = build_parser().parse_args(argv)

    draws = draws_factory(args.family, args.config_root)
    written = generate(draws, args.output_root, args.n_shuffles, max_workers=args.max_workers)
    report_written(written)


if __name__ == "__main__":
    main()
