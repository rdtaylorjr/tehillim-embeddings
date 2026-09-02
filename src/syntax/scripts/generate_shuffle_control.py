"""Generates N within-half-verse-shuffled datasets: a shuffle-null for an ordered rep."""

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
from syntax.corpus import Corpus, PhrasePsalm
from syntax.function_ngram import (
    phrase_function_1_2_3gram_psalm_sparse_vectors,
    phrase_function_1_2_3gram_sparse_vectors,
    phrase_function_1_2gram_psalm_vectors,
    phrase_function_1_2gram_vectors,
)
from syntax.signature_support import MIN_EXTERNAL_SUPPORT_K
from syntax.signature_vectorize import (
    phrase_signature_1_2_3gram_psalm_sparse_vectors,
    phrase_signature_1_2_3gram_sparse_vectors,
    phrase_signature_1_2gram_psalm_vectors,
    phrase_signature_1_2gram_vectors,
)
from syntax.typ_ngram import (
    phrase_typ_1_2_3gram_psalm_sparse_vectors,
    phrase_typ_1_2_3gram_sparse_vectors,
    phrase_typ_1_2gram_psalm_vectors,
    phrase_typ_1_2gram_vectors,
)
from syntax.vocabulary import FUNCTION_VOCABULARY, TYP_VOCABULARY

_DATASET_TYPE = "syntax"

_DIM_BY_UNIT = {"typ": len(TYP_VOCABULARY), "function": len(FUNCTION_VOCABULARY)}

_DENSE_SIGNATURE_BUILDERS = {
    "1_2gram": phrase_signature_1_2gram_vectors,
    "1_2gram_psalm": phrase_signature_1_2gram_psalm_vectors,
}

#: The trigram block is 99.97% zeros at these dimensions, so it is stored sparsely.
_SPARSE_SIGNATURE_BUILDERS = {
    "1_2_3gram": phrase_signature_1_2_3gram_sparse_vectors,
    "1_2_3gram_psalm": phrase_signature_1_2_3gram_psalm_sparse_vectors,
}

_DENSE_BUILDERS_BY_UNIT = {
    "typ": {
        "1_2gram": phrase_typ_1_2gram_vectors,
        "1_2gram_psalm": phrase_typ_1_2gram_psalm_vectors,
    },
    "function": {
        "1_2gram": phrase_function_1_2gram_vectors,
        "1_2gram_psalm": phrase_function_1_2gram_psalm_vectors,
    },
}

#: The trigram block is 99.97% zeros at these dimensions, so it is stored sparsely.
_SPARSE_BUILDERS_BY_UNIT = {
    "typ": {
        "1_2_3gram": phrase_typ_1_2_3gram_sparse_vectors,
        "1_2_3gram_psalm": phrase_typ_1_2_3gram_psalm_sparse_vectors,
    },
    "function": {
        "1_2_3gram": phrase_function_1_2_3gram_sparse_vectors,
        "1_2_3gram_psalm": phrase_function_1_2_3gram_psalm_sparse_vectors,
    },
}


def _half_verse_typ(psalm: PhrasePsalm) -> tuple[tuple[str, ...], ...]:
    """Selects the phrase-atom sequence whose order the shuffle-null permutes."""
    return psalm.half_verse_typ


@dataclass(frozen=True, slots=True)
class _UnitContext:
    """Everything one unit seed needs, pickled once per worker process rather than once per seed."""

    psalms: tuple[PhrasePsalm, ...]
    output_root: Path
    unit: str
    representation: str


@dataclass(frozen=True, slots=True)
class _SignatureContext:
    """Everything one signature seed needs, pickled once per worker rather than once per seed."""

    psalms: tuple[PhrasePsalm, ...]
    output_root: Path
    representation: str
    vocabulary: tuple[str, ...]
    external_counts: dict[str, int]
    k: int


def write_unit_seed(context: _UnitContext, seed: int) -> str:
    """Writes one seeded, within-half-verse-shuffled `<unit>_<representation>` dataset."""
    psalms = list(context.psalms)
    order = shuffled_within_half_verse_order(psalms, seed, half_verses=_half_verse_typ)
    name = shuffle_construction_name(context.representation, seed)
    description = (
        f"Shuffle-null order-effect control for {context.unit}_{context.representation}, "
        f"seed {seed}."
    )
    dense_builder = _DENSE_BUILDERS_BY_UNIT[context.unit].get(context.representation)
    if dense_builder is not None:
        write_dataset(
            context.output_root,
            context.unit,
            name,
            dense_builder(psalms, order),
            description,
            domain=_DATASET_TYPE,
            unit_key="feature",
            level="phrase",
        )
    else:
        sparse_builder = _SPARSE_BUILDERS_BY_UNIT[context.unit][context.representation]
        dim = _DIM_BY_UNIT[context.unit]
        write_sparse_dataset(
            context.output_root,
            context.unit,
            name,
            sparse_builder(psalms, order),
            dim + dim * dim + dim * dim * dim,
            description,
            domain=_DATASET_TYPE,
            unit_key="feature",
            level="phrase",
        )
    return f"{context.unit}_{name}"


def write_signature_seed(context: _SignatureContext, seed: int) -> str:
    """Writes one seeded, within-half-verse-shuffled `phrase_signature_<representation>` dataset."""
    psalms = list(context.psalms)
    order = shuffled_within_half_verse_order(psalms, seed, half_verses=_half_verse_typ)
    name = shuffle_construction_name(context.representation, seed)
    description = (
        f"Shuffle-null order-effect control for phrase_signature_{context.representation}, "
        f"seed {seed}."
    )
    args = (psalms, context.vocabulary, context.external_counts, context.k, order)
    dense_builder = _DENSE_SIGNATURE_BUILDERS.get(context.representation)
    if dense_builder is not None:
        write_dataset(
            context.output_root,
            "signature",
            name,
            dense_builder(*args),
            description,
            domain=_DATASET_TYPE,
            unit_key="feature",
            level="phrase",
        )
    else:
        dim = len(context.vocabulary)
        write_sparse_dataset(
            context.output_root,
            "signature",
            name,
            _SPARSE_SIGNATURE_BUILDERS[context.representation](*args),
            dim + dim * dim + dim * dim * dim,
            description,
            domain=_DATASET_TYPE,
            unit_key="feature",
            level="phrase",
        )
    return f"signature_{name}"


def generate_shuffle_control(
    psalms: list[PhrasePsalm],
    output_root: Path,
    unit: str,
    representation: str,
    n_shuffles: int,
    *,
    max_workers: int | None = None,
) -> list[str]:
    """Writes n_shuffles seeded, within-half-verse-shuffled `<unit>_<representation>` datasets."""
    builders = {**_DENSE_BUILDERS_BY_UNIT[unit], **_SPARSE_BUILDERS_BY_UNIT[unit]}
    if representation not in builders:
        raise ValueError(
            f"representation {representation!r} has no shuffle control "
            f"(unigram histograms are provably order-invariant); "
            f"choose one of {sorted(builders)}"
        )
    context = _UnitContext(
        psalms=tuple(psalms),
        output_root=output_root,
        unit=unit,
        representation=representation,
    )
    return map_seeds(write_unit_seed, context, range(1, n_shuffles + 1), max_workers=max_workers)


def generate_signature_shuffle_control(
    psalms: list[PhrasePsalm],
    output_root: Path,
    representation: str,
    n_shuffles: int,
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
    *,
    max_workers: int | None = None,
) -> list[str]:
    """Writes n_shuffles seeded within-half-verse `phrase_signature_<rep>` datasets."""
    if representation not in {**_DENSE_SIGNATURE_BUILDERS, **_SPARSE_SIGNATURE_BUILDERS}:
        raise ValueError(
            f"representation {representation!r} has no shuffle control; "
            f"choose one of {sorted({**_DENSE_SIGNATURE_BUILDERS, **_SPARSE_SIGNATURE_BUILDERS})}"
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
    """Command-line interface for the phrase unit and signature shuffle-null controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unit", required=True, choices=(*sorted(_DENSE_BUILDERS_BY_UNIT), "signature")
    )
    parser.add_argument("--representation", required=True)
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
    """Generates the shuffle-null control datasets for one phrase representation."""
    args = build_parser().parse_args(argv)

    corpus = corpus_factory()
    psalms = corpus.psalms()
    if args.unit == "signature":
        support_path = args.config_root / "phrase_signature_external_support.csv"
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
    else:
        written = generate_shuffle_control(
            psalms,
            args.output_root,
            args.unit,
            args.representation,
            args.n_shuffles,
            max_workers=args.max_workers,
        )
    print(f"wrote {len(written)} shuffle-control datasets", file=sys.stderr)


if __name__ == "__main__":
    main()
