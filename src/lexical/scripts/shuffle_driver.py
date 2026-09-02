"""Shared per-seed driver for the lexical half-verse-order shuffle-null controls."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.columns import PsalmColumns
from core.export import write_dataset
from core.parallel import map_seeds
from core.shuffle import DEFAULT_N_SHUFFLES, shuffle_construction_name, shuffled_order_by_psalm
from lexical.corpus import Corpus, LexicalPsalm
from lexical.frequency import icf_weights as compute_icf_weights
from lexical.frequency import lex0_token_frequencies, total_token_count
from lexical.vocabulary import build_vocabulary, columns_for_key

#: Builds one seed's vectors under a half-verse-order permutation. Must be module-level to pickle.
VectorBuilder = Callable[
    [list[PsalmColumns], tuple[str, ...], dict[str, float], dict[int, np.ndarray]],
    dict[int, np.ndarray],
]


@dataclass(frozen=True, slots=True)
class SeedContext:
    """Everything one seed needs, pickled once per worker process rather than once per seed."""

    psalms: tuple[LexicalPsalm, ...]
    output_root: Path
    vocabulary: tuple[str, ...]
    icf_weights: dict[str, float]
    construction: str
    builder: VectorBuilder


def write_seed(context: SeedContext, seed: int) -> str:
    """Writes one seeded, half-verse-order-shuffled dataset and returns its construction name."""
    psalms = list(context.psalms)
    #: The builder applies the permutation: it moves a half-verse's position, never its content.
    order = shuffled_order_by_psalm(psalms, seed)
    columns = columns_for_key(psalms, "lex0")
    vectors = context.builder(columns, context.vocabulary, context.icf_weights, order)
    name = shuffle_construction_name(context.construction, seed)
    description = f"Shuffle-null order-effect control for {context.construction}, seed {seed}."
    write_dataset(context.output_root, "homograph", name, vectors, description, unit_key="unit")
    return name


def generate(
    psalms: list[LexicalPsalm],
    output_root: Path,
    icf_weights: dict[str, float],
    n_shuffles: int,
    *,
    construction: str,
    builder: VectorBuilder,
    max_workers: int | None = None,
) -> list[str]:
    """Writes n_shuffles seeded, half-verse-order-shuffled datasets, returns the names written."""
    context = SeedContext(
        psalms=tuple(psalms),
        output_root=output_root,
        vocabulary=build_vocabulary(psalms, key="lex0"),
        icf_weights=icf_weights,
        construction=construction,
        builder=builder,
    )
    return map_seeds(write_seed, context, range(1, n_shuffles + 1), max_workers=max_workers)


def build_parser(doc: str | None) -> argparse.ArgumentParser:
    """Command-line interface shared by both lexical shuffle-null controls."""
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES)
    parser.add_argument("--max-workers", type=int, default=None)
    return parser


def run(
    doc: str | None,
    construction: str,
    builder: VectorBuilder,
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus] = Corpus.load,
) -> None:
    """Loads the corpus, generates every seed, and reports how many datasets were written."""
    args = build_parser(doc).parse_args(argv)
    corpus = corpus_factory()
    psalms = corpus.psalms()
    icf_lookup = compute_icf_weights(
        lex0_token_frequencies(corpus.api), total_token_count(corpus.api)
    )
    written = generate(
        psalms,
        args.output_root,
        icf_lookup,
        args.n_shuffles,
        construction=construction,
        builder=builder,
        max_workers=args.max_workers,
    )
    print(f"wrote {len(written)} shuffle-control datasets", file=sys.stderr)
