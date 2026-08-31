"""Generates N within-colon-shuffled datasets: a shuffle-null control for an ordered rep."""

from __future__ import annotations

import sys
from pathlib import Path

from lexical.export import write_dataset, write_sparse_dataset
from lexical.shuffle_control import DEFAULT_N_SHUFFLES
from morphology.corpus import Corpus, MorphologicalPsalm
from morphology.pos_ngram import (
    sp_1_2_3gram_psalm_vectors,
    sp_1_2_3gram_vectors,
    sp_1_2gram_psalm_vectors,
    sp_1_2gram_vectors,
)
from morphology.shuffle_control import shuffled_within_colon_order
from morphology.signature_support import (
    MIN_EXTERNAL_SUPPORT_K,
    build_signature_vocabulary,
    load_external_signature_counts,
)
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


def generate_shuffle_control(
    psalms: list[MorphologicalPsalm],
    output_root: Path,
    representation: str,
    n_shuffles: int,
) -> list[str]:
    """Writes n_shuffles seeded, within-colon-order-shuffled `sp_<representation>` datasets."""
    builder = _POS_BUILDERS.get(representation)
    if builder is None:
        raise ValueError(
            f"representation {representation!r} has no shuffle control "
            f"(unigram histograms are provably order-invariant); "
            f"choose one of {sorted(_POS_BUILDERS)}"
        )

    written: list[str] = []
    for seed in range(1, n_shuffles + 1):
        order = shuffled_within_colon_order(psalms, seed)
        vectors = builder(psalms, order)
        weight = f"{representation}_shuffle{seed:02d}"
        description = f"Shuffle-null order-effect control for sp_{representation}, seed {seed}."
        write_dataset(
            output_root,
            "sp",
            weight,
            vectors,
            description,
            domain=_DATASET_TYPE,
            unit_key="feature",
        )
        written.append(f"sp_{weight}")
    return written


def generate_signature_shuffle_control(
    psalms: list[MorphologicalPsalm],
    output_root: Path,
    representation: str,
    n_shuffles: int,
    vocabulary: tuple[str, ...],
    external_counts: dict[str, int],
    k: int,
) -> list[str]:
    """Writes n_shuffles seeded, within-colon-order-shuffled `morph_signature_<rep>` datasets."""
    dim = len(vocabulary)
    dense_builder = _DENSE_SIGNATURE_BUILDERS.get(representation)
    sparse_builder = _SPARSE_SIGNATURE_BUILDERS.get(representation)
    if dense_builder is None and sparse_builder is None:
        raise ValueError(
            f"representation {representation!r} has no shuffle control "
            f"(unigram histograms are provably order-invariant); choose one of "
            f"{sorted(_DENSE_SIGNATURE_BUILDERS) + sorted(_SPARSE_SIGNATURE_BUILDERS)}"
        )

    written: list[str] = []
    for seed in range(1, n_shuffles + 1):
        order = shuffled_within_colon_order(psalms, seed)
        weight = f"{representation}_shuffle{seed:02d}"
        description = (
            f"Shuffle-null order-effect control for morph_signature_{representation}, seed {seed}."
        )
        if dense_builder is not None:
            vectors = dense_builder(psalms, vocabulary, external_counts, k, order)
            write_dataset(
                output_root,
                "morph_signature",
                weight,
                vectors,
                description,
                domain=_DATASET_TYPE,
                unit_key="feature",
            )
        else:
            assert sparse_builder is not None
            sparse_vectors = sparse_builder(psalms, vocabulary, external_counts, k, order)
            write_sparse_dataset(
                output_root,
                "morph_signature",
                weight,
                sparse_vectors,
                dim + dim * dim + dim * dim * dim,
                description,
                domain=_DATASET_TYPE,
                unit_key="feature",
            )
        written.append(f"morph_signature_{weight}")
    return written


def main() -> None:
    """Generates the shuffle-null control datasets for one representation."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=("pos", "signature"))
    parser.add_argument("--representation", required=True, choices=sorted(_POS_BUILDERS))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES)
    args = parser.parse_args()

    corpus = Corpus.load()
    psalms = corpus.psalms()

    if args.family == "pos":
        written = generate_shuffle_control(
            psalms, args.output_root, args.representation, args.n_shuffles
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
        )
    print(f"wrote {len(written)} shuffle-control datasets", file=sys.stderr)


if __name__ == "__main__":
    main()
