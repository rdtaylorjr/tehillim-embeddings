"""Computes and writes morph_atomic (dim 66) and the morph_signature family of constructions."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from lexical.export import dataset_path, write_dataset, write_sparse_dataset
from morphological.corpus import Corpus, MorphologicalPsalm
from morphological.signature_support import (
    MIN_EXTERNAL_SUPPORT_K,
    build_signature_vocabulary,
    load_external_signature_counts,
)
from morphological.signature_vectorize import (
    morph_atomic_psalm_vectors,
    morph_atomic_vectors,
    morph_signature_1_2_3gram_psalm_sparse_vectors,
    morph_signature_1_2_3gram_sparse_vectors,
    morph_signature_1_2gram_psalm_vectors,
    morph_signature_1_2gram_vectors,
    morph_signature_psalm_vectors,
    morph_signature_vectors,
)

_DATASET_TYPE = "morphological"

_DEFAULT_SUPPORT_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "config" / "morph_signature_external_support.csv"
)


def generate(
    psalms: list[MorphologicalPsalm],
    output_root: Path,
    external_counts: dict[str, int],
    k: int,
) -> list[str]:
    """Writes every not-yet-written morph_atomic/morph_signature construction, returns names."""
    written: list[str] = []

    for construction, builder in (
        ("core", morph_atomic_vectors),
        ("core_psalm", morph_atomic_psalm_vectors),
    ):
        if dataset_path(
            output_root, "morph_atomic", construction, dataset_type=_DATASET_TYPE
        ).exists():
            continue
        print(
            f"computing morphological unit=morph_atomic construction={construction}...",
            file=sys.stderr,
        )
        vectors = builder(psalms)
        description = (
            f"Atomic morphology baseline [sp;gn;nu;ps;st;vs;vt], construction={construction}."
        )
        write_dataset(
            output_root,
            "morph_atomic",
            construction,
            vectors,
            description,
            dataset_type=_DATASET_TYPE,
        )
        written.append(f"morph_atomic_{construction}")

    vocabulary = build_signature_vocabulary(external_counts, k)
    dim = len(vocabulary)

    dense_builders: dict[str, Callable[[], dict[int, np.ndarray]]] = {
        "inventory": lambda: morph_signature_vectors(psalms, vocabulary, external_counts, k),
        "inventory_psalm": lambda: morph_signature_psalm_vectors(
            psalms, vocabulary, external_counts, k
        ),
        "1_2gram": lambda: morph_signature_1_2gram_vectors(psalms, vocabulary, external_counts, k),
        "1_2gram_psalm": lambda: morph_signature_1_2gram_psalm_vectors(
            psalms, vocabulary, external_counts, k
        ),
    }
    for construction, signature_builder in dense_builders.items():
        if dataset_path(
            output_root, "morph_signature", construction, dataset_type=_DATASET_TYPE
        ).exists():
            continue
        print(
            f"computing morphological unit=morph_signature construction={construction}...",
            file=sys.stderr,
        )
        description = (
            f"Grammatical-signature histogram (RARE-collapsed, k={k}), construction={construction}."
        )
        write_dataset(
            output_root,
            "morph_signature",
            construction,
            signature_builder(),
            description,
            dataset_type=_DATASET_TYPE,
        )
        written.append(f"morph_signature_{construction}")

    # 1_2_3gram's dim (42^1 + 42^2 + 42^3) is huge and almost entirely zero per colon (at most
    # ~8 nonzero entries), so it is stored sparsely rather than as a dense array.
    combined_dim = dim + dim * dim + dim * dim * dim
    sparse_builders: dict[str, Callable[[], dict[int, tuple[np.ndarray, np.ndarray]]]] = {
        "1_2_3gram": lambda: morph_signature_1_2_3gram_sparse_vectors(
            psalms, vocabulary, external_counts, k
        ),
        "1_2_3gram_psalm": lambda: morph_signature_1_2_3gram_psalm_sparse_vectors(
            psalms, vocabulary, external_counts, k
        ),
    }
    for construction, sparse_builder in sparse_builders.items():
        if dataset_path(
            output_root, "morph_signature", construction, dataset_type=_DATASET_TYPE
        ).exists():
            continue
        print(
            f"computing morphological unit=morph_signature construction={construction}...",
            file=sys.stderr,
        )
        description = (
            f"Grammatical-signature histogram (RARE-collapsed, k={k}), construction={construction}."
        )
        write_sparse_dataset(
            output_root,
            "morph_signature",
            construction,
            sparse_builder(),
            combined_dim,
            description,
            dataset_type=_DATASET_TYPE,
        )
        written.append(f"morph_signature_{construction}")

    return written


def main() -> None:
    """Generates every missing morph_atomic/morph_signature dataset."""
    output_root = Path(__file__).resolve().parents[2]
    corpus = Corpus.load()
    psalms = corpus.psalms()
    external_counts = load_external_signature_counts(_DEFAULT_SUPPORT_PATH)
    written = generate(psalms, output_root, external_counts, MIN_EXTERNAL_SUPPORT_K)
    print(f"wrote {len(written)} dataset files", file=sys.stderr)


if __name__ == "__main__":
    main()
