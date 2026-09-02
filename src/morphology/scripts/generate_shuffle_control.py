"""Generates N within-half-verse-shuffled datasets: a shuffle-null control for an ordered rep."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core.export import write_dataset, write_sparse_dataset
from core.parallel import map_seeds
from core.shuffle import (
    DEFAULT_N_SHUFFLES,
    shuffle_construction_name,
    shuffled_within_half_verse_order,
)
from core.support import build_signature_vocabulary, load_external_signature_counts
from morphology.corpus import Corpus, MorphologicalPsalm
from morphology.pos_ngram import (
    sp_1_2_3gram_psalm_vectors,
    sp_1_2_3gram_vectors,
    sp_1_2gram_psalm_vectors,
    sp_1_2gram_vectors,
)
from morphology.signature_support import MIN_EXTERNAL_SUPPORT_K
from morphology.signature_vectorize import (
    morph_signature_1_2_3gram_psalm_sparse_vectors,
    morph_signature_1_2_3gram_sparse_vectors,
    morph_signature_1_2gram_psalm_vectors,
    morph_signature_1_2gram_vectors,
)

_DATASET_TYPE = "morphology"

_POS_BUILDERS = {
    "1_2gram": sp_1_2gram_vectors,
    "1_2_3gram": sp_1_2_3gram_vectors,
    "1_2gram_psalm": sp_1_2gram_psalm_vectors,
    "1_2_3gram_psalm": sp_1_2_3gram_psalm_vectors,
}

_DENSE_SIGNATURE_BUILDERS = {
    "1_2gram": morph_signature_1_2gram_vectors,
    "1_2gram_psalm": morph_signature_1_2gram_psalm_vectors,
}

_SPARSE_SIGNATURE_BUILDERS = {
    "1_2_3gram": morph_signature_1_2_3gram_sparse_vectors,
    "1_2_3gram_psalm": morph_signature_1_2_3gram_psalm_sparse_vectors,
}


def _half_verse_sp(psalm: MorphologicalPsalm) -> tuple[tuple[str, ...], ...]:
    """Selects the word-level sequence whose order the shuffle-null permutes."""
    return psalm.half_verse_sp


@dataclass(frozen=True, slots=True)
class _PosContext:
    """Everything one POS seed needs, pickled once per worker process rather than once per seed."""

    psalms: tuple[MorphologicalPsalm, ...]
    output_root: Path
    representation: str


@dataclass(frozen=True, slots=True)
class _SignatureContext:
    """Everything one signature seed needs, pickled once per worker rather than once per seed."""

    psalms: tuple[MorphologicalPsalm, ...]
    output_root: Path
    representation: str
    vocabulary: tuple[str, ...]
    external_counts: dict[str, int]
    k: int


def write_pos_seed(context: _PosContext, seed: int) -> str:
    """Writes one seeded, within-half-verse-order-shuffled `sp_<representation>` dataset."""
    psalms = list(context.psalms)
    order = shuffled_within_half_verse_order(psalms, seed, half_verses=_half_verse_sp)
    vectors = _POS_BUILDERS[context.representation](psalms, order)
    name = shuffle_construction_name(context.representation, seed)
    description = f"Shuffle-null order-effect control for sp_{context.representation}, seed {seed}."
    write_dataset(
        context.output_root,
        "sp",
        name,
        vectors,
        description,
        domain=_DATASET_TYPE,
        unit_key="feature",
    )
    return f"sp_{name}"


def write_signature_seed(context: _SignatureContext, seed: int) -> str:
    """Writes one seeded, within-half-verse-order-shuffled `morph_signature_<rep>` dataset."""
    psalms = list(context.psalms)
    dim = len(context.vocabulary)
    order = shuffled_within_half_verse_order(psalms, seed, half_verses=_half_verse_sp)
    name = shuffle_construction_name(context.representation, seed)
    description = (
        f"Shuffle-null order-effect control for morph_signature_{context.representation}, "
        f"seed {seed}."
    )
    dense_builder = _DENSE_SIGNATURE_BUILDERS.get(context.representation)
    if dense_builder is not None:
        vectors = dense_builder(
            psalms, context.vocabulary, context.external_counts, context.k, order
        )
        write_dataset(
            context.output_root,
            "morph_signature",
            name,
            vectors,
            description,
            domain=_DATASET_TYPE,
            unit_key="feature",
        )
    else:
        sparse_vectors = _SPARSE_SIGNATURE_BUILDERS[context.representation](
            psalms, context.vocabulary, context.external_counts, context.k, order
        )
        write_sparse_dataset(
            context.output_root,
            "morph_signature",
            name,
            sparse_vectors,
            dim + dim * dim + dim * dim * dim,
            description,
            domain=_DATASET_TYPE,
            unit_key="feature",
        )
    return f"morph_signature_{name}"


def generate_shuffle_control(
    psalms: list[MorphologicalPsalm],
    output_root: Path,
    representation: str,
    n_shuffles: int,
    *,
    max_workers: int | None = None,
) -> list[str]:
    """Writes n_shuffles seeded, within-half-verse-order-shuffled `sp_<representation>` datasets."""
    if representation not in _POS_BUILDERS:
        raise ValueError(
            f"representation {representation!r} has no shuffle control "
            f"(unigram histograms are provably order-invariant); "
            f"choose one of {sorted(_POS_BUILDERS)}"
        )
    context = _PosContext(
        psalms=tuple(psalms), output_root=output_root, representation=representation
    )
    return map_seeds(write_pos_seed, context, range(1, n_shuffles + 1), max_workers=max_workers)


def generate_signature_shuffle_control(
    psalms: list[MorphologicalPsalm],
    output_root: Path,
    representation: str,
    n_shuffles: int,
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    *,
    max_workers: int | None = None,
) -> list[str]:
    """Writes n_shuffles seeded, within-order-shuffled `morph_signature_<rep>` datasets."""
    if (
        representation not in _DENSE_SIGNATURE_BUILDERS
        and representation not in _SPARSE_SIGNATURE_BUILDERS
    ):
        raise ValueError(
            f"representation {representation!r} has no shuffle control "
            f"(unigram histograms are provably order-invariant); choose one of "
            f"{sorted(_DENSE_SIGNATURE_BUILDERS) + sorted(_SPARSE_SIGNATURE_BUILDERS)}"
        )
    context = _SignatureContext(
        psalms=tuple(psalms),
        output_root=output_root,
        representation=representation,
        vocabulary=vocabulary,
        external_counts=external_counts,
        k=k,
    )
    return map_seeds(
        write_signature_seed, context, range(1, n_shuffles + 1), max_workers=max_workers
    )


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for the POS and signature shuffle-null controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=("pos", "signature"))
    parser.add_argument("--representation", required=True, choices=sorted(_POS_BUILDERS))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES)
    parser.add_argument("--max-workers", type=int, default=None)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus] = Corpus.load,
) -> None:
    """Generates the shuffle-null control datasets for one representation."""
    args = build_parser().parse_args(argv)

    corpus = corpus_factory()
    psalms = corpus.psalms()

    if args.family == "pos":
        written = generate_shuffle_control(
            psalms,
            args.output_root,
            args.representation,
            args.n_shuffles,
            max_workers=args.max_workers,
        )
    else:
        support_path = args.config_root / "morph_signature_external_support.csv"
        external_counts = load_external_signature_counts(support_path)
        vocabulary = build_signature_vocabulary(external_counts, MIN_EXTERNAL_SUPPORT_K)
        written = generate_signature_shuffle_control(
            psalms,
            args.output_root,
            args.representation,
            args.n_shuffles,
            vocabulary,
            external_counts,
            MIN_EXTERNAL_SUPPORT_K,
            max_workers=args.max_workers,
        )
    print(f"wrote {len(written)} shuffle-control datasets", file=sys.stderr)


if __name__ == "__main__":
    main()
