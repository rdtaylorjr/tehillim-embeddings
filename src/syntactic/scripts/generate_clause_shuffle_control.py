"""Generates N within-colon-shuffled clause datasets: the shuffle-null for the ordered families."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core.cli import (
    add_config_root_argument,
    add_output_root_argument,
    add_shuffle_arguments,
    report_written,
)
from core.export import write_dataset, write_sparse_dataset
from core.ngram import concatenated_1_2_3gram_dim
from core.parallel import map_seeds
from core.shuffle import shuffle_construction_name, shuffled_within_half_verse_order
from syntactic import DATASET_TYPE
from syntactic.clause_ordered import (
    ORDERED_UNITS,
    OrderedFamily,
    dense_vectors,
    resolve_family,
    sparse_vectors,
)
from syntactic.corpus import ClausePsalm, Corpus, clause_corpus


@dataclass(frozen=True, slots=True)
class _Context:
    """Everything one seed needs, pickled once per worker rather than once per draw."""

    psalms: tuple[ClausePsalm, ...]
    output_root: Path
    family: OrderedFamily


def write_seed(context: _Context, seed: int) -> list[str]:
    """Writes every ordered construction for one shuffle seed, returns the names written."""
    family = context.family
    psalms = list(context.psalms)
    order = shuffled_within_half_verse_order(psalms, seed, half_verses=family.columns_of)
    written: list[str] = []
    for construction in family.dense:
        name = shuffle_construction_name(construction, seed)
        write_dataset(
            context.output_root,
            family.unit,
            name,
            dense_vectors(family, psalms, construction, order),
            f"{family.unit} {construction}, within-colon order shuffle seed {seed}.",
            unit_key="feature",
            level="clause",
            domain=DATASET_TYPE,
        )
        written.append(f"{family.unit}_{name}")
    dim = concatenated_1_2_3gram_dim(len(family.vocabulary))
    for construction in family.sparse:
        name = shuffle_construction_name(construction, seed)
        write_sparse_dataset(
            context.output_root,
            family.unit,
            name,
            sparse_vectors(family, psalms, construction, order),
            dim,
            f"{family.unit} {construction}, within-colon order shuffle seed {seed}.",
            unit_key="feature",
            level="clause",
            domain=DATASET_TYPE,
        )
        written.append(f"{family.unit}_{name}")
    return written


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus[ClausePsalm]] = clause_corpus,
) -> None:
    """Generates every shuffle draw for one clause family."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_output_root_argument(parser)
    add_config_root_argument(parser)
    add_shuffle_arguments(parser)
    parser.add_argument("--unit", choices=ORDERED_UNITS, required=True)
    args = parser.parse_args(argv)
    context = _Context(
        psalms=tuple(corpus_factory().psalms()),
        output_root=args.output_root,
        family=resolve_family(args.unit, args.config_root),
    )
    seeds = range(1, args.n_shuffles + 1)
    written = map_seeds(write_seed, context, seeds, max_workers=args.max_workers)
    report_written([name for names in written for name in names])


if __name__ == "__main__":
    main()
