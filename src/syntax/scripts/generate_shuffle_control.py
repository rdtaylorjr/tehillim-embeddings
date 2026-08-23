"""Generates N within-colon-shuffled datasets: a shuffle-null control for an ordered phrase rep."""

from __future__ import annotations

import sys
from pathlib import Path

from lexical.export import write_dataset
from syntax.corpus import Corpus, PhrasePsalm
from syntax.function_ngram import (
    phrase_function_1_2_3gram_psalm_vectors,
    phrase_function_1_2_3gram_vectors,
    phrase_function_1_2gram_psalm_vectors,
    phrase_function_1_2gram_vectors,
)
from syntax.shuffle_control import shuffled_within_colon_order
from syntax.signature_support import (
    MIN_EXTERNAL_SUPPORT_K,
    build_signature_vocabulary,
    load_external_signature_counts,
)
from syntax.signature_vectorize import (
    phrase_signature_1_2_3gram_psalm_vectors,
    phrase_signature_1_2_3gram_vectors,
    phrase_signature_1_2gram_psalm_vectors,
    phrase_signature_1_2gram_vectors,
)
from syntax.typ_ngram import (
    phrase_typ_1_2_3gram_psalm_vectors,
    phrase_typ_1_2_3gram_vectors,
    phrase_typ_1_2gram_psalm_vectors,
    phrase_typ_1_2gram_vectors,
)

_DATASET_TYPE = "syntax"

_SIGNATURE_BUILDERS = {
    "1_2gram": phrase_signature_1_2gram_vectors,
    "1_2_3gram": phrase_signature_1_2_3gram_vectors,
    "1_2gram_psalm": phrase_signature_1_2gram_psalm_vectors,
    "1_2_3gram_psalm": phrase_signature_1_2_3gram_psalm_vectors,
}

_BUILDERS_BY_UNIT = {
    "typ": {
        "1_2gram": phrase_typ_1_2gram_vectors,
        "1_2_3gram": phrase_typ_1_2_3gram_vectors,
        "1_2gram_psalm": phrase_typ_1_2gram_psalm_vectors,
        "1_2_3gram_psalm": phrase_typ_1_2_3gram_psalm_vectors,
    },
    "function": {
        "1_2gram": phrase_function_1_2gram_vectors,
        "1_2_3gram": phrase_function_1_2_3gram_vectors,
        "1_2gram_psalm": phrase_function_1_2gram_psalm_vectors,
        "1_2_3gram_psalm": phrase_function_1_2_3gram_psalm_vectors,
    },
}


def generate_shuffle_control(
    psalms: list[PhrasePsalm], output_root: Path, unit: str, representation: str, n_shuffles: int
) -> list[str]:
    """Writes n_shuffles seeded, within-colon-shuffled `<unit>_<representation>` datasets."""
    builders = _BUILDERS_BY_UNIT[unit]
    builder = builders.get(representation)
    if builder is None:
        raise ValueError(
            f"representation {representation!r} has no shuffle control "
            f"(unigram histograms are provably order-invariant); "
            f"choose one of {sorted(builders)}"
        )

    written: list[str] = []
    for seed in range(1, n_shuffles + 1):
        order = shuffled_within_colon_order(psalms, seed)
        vectors = builder(psalms, order)
        weight = f"{representation}_shuffle{seed:02d}"
        description = f"Shuffle-null order-effect control for {unit}_{representation}, seed {seed}."
        write_dataset(
            output_root,
            unit,
            weight,
            vectors,
            description,
            domain=_DATASET_TYPE,
            unit_key="feature",
            level="phrase",
        )
        written.append(f"{unit}_{weight}")
    return written


def generate_signature_shuffle_control(
    psalms: list[PhrasePsalm],
    output_root: Path,
    representation: str,
    n_shuffles: int,
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
) -> list[str]:
    """Writes n_shuffles seeded, within-colon-shuffled `phrase_signature_<representation>` sets."""
    builder = _SIGNATURE_BUILDERS.get(representation)
    if builder is None:
        raise ValueError(
            f"representation {representation!r} has no shuffle control; "
            f"choose one of {sorted(_SIGNATURE_BUILDERS)}"
        )

    written: list[str] = []
    for seed in range(1, n_shuffles + 1):
        order = shuffled_within_colon_order(psalms, seed)
        vectors = builder(psalms, vocabulary, external_counts, k, order)
        weight = f"{representation}_shuffle{seed:02d}"
        description = (
            f"Shuffle-null order-effect control for phrase_signature_{representation}, seed {seed}."
        )
        write_dataset(
            output_root,
            "signature",
            weight,
            vectors,
            description,
            domain=_DATASET_TYPE,
            unit_key="feature",
            level="phrase",
        )
        written.append(f"signature_{weight}")
    return written


def main() -> None:
    """Generates the shuffle-null control datasets for one phrase representation."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", required=True, choices=(*sorted(_BUILDERS_BY_UNIT), "signature"))
    parser.add_argument("--representation", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--n-shuffles", type=int, default=30)
    args = parser.parse_args()

    corpus = Corpus.load()
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
        )
    else:
        written = generate_shuffle_control(
            psalms, args.output_root, args.unit, args.representation, args.n_shuffles
        )
    print(f"wrote {len(written)} shuffle-control datasets", file=sys.stderr)


if __name__ == "__main__":
    main()
