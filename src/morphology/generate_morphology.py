"""Computes and writes the atomic-morphology family: 9 features, atomic and sp+feature."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from core.cli import run_generator
from core.export import dataset_path, write_dataset
from morphology import DATASET_TYPE
from morphology.atomic import (
    FeatureKey,
    atomic_psalm_vectors,
    atomic_vectors,
    full_morphology_psalm_vectors,
    full_morphology_vectors,
    sp_plus_feature_psalm_vectors,
    sp_plus_feature_vectors,
)
from morphology.corpus import Corpus, MorphologicalPsalm

_FEATURES: tuple[FeatureKey, ...] = (
    "gn",
    "nu",
    "ps",
    "st",
    "vs",
    "vt",
    "prs_gn",
    "prs_nu",
    "prs_ps",
)


def _write_if_missing(
    output_root: Path,
    unit: str,
    construction: str,
    vectors: dict[int, np.ndarray],
    description: str,
) -> bool:
    """Writes one construction unless its Parquet file already exists, reporting whether it did."""
    if dataset_path(
        output_root, unit, construction, domain=DATASET_TYPE, unit_key="feature"
    ).exists():
        return False
    write_dataset(
        output_root,
        unit,
        construction,
        vectors,
        description,
        domain=DATASET_TYPE,
        unit_key="feature",
    )
    return True


def generate(psalms: list[MorphologicalPsalm], output_root: Path) -> list[str]:
    """Writes every not-yet-written atomic-morphology construction, returns the names written."""
    written: list[str] = []

    for feature in _FEATURES:
        unit = f"morph_{feature}"

        for construction, builder in (
            ("atomic", atomic_vectors),
            ("sp_plus", sp_plus_feature_vectors),
        ):
            print(
                f"computing morphology feature={unit} construction={construction}...",
                file=sys.stderr,
            )
            if _write_if_missing(
                output_root,
                unit,
                construction,
                builder(psalms, feature),
                f"Atomic morphology histogram for {feature}, construction={construction}.",
            ):
                written.append(f"{unit}_{construction}")

        for construction, builder in (
            ("atomic_psalm", atomic_psalm_vectors),
            ("sp_plus_psalm", sp_plus_feature_psalm_vectors),
        ):
            print(
                f"computing morphology feature={unit} construction={construction}...",
                file=sys.stderr,
            )
            if _write_if_missing(
                output_root,
                unit,
                construction,
                builder(psalms, feature),
                f"Atomic morphology histogram for {feature}, construction={construction}.",
            ):
                written.append(f"{unit}_{construction}")

    print("computing morphology feature=morph_full construction=all...", file=sys.stderr)
    if _write_if_missing(
        output_root,
        "morph_full",
        "all",
        full_morphology_vectors(psalms),
        "Full morphology baseline: sp + all 9 atomic features concatenated, dim 77.",
    ):
        written.append("morph_full_all")

    print("computing morphology feature=morph_full construction=all_psalm...", file=sys.stderr)
    if _write_if_missing(
        output_root,
        "morph_full",
        "all_psalm",
        full_morphology_psalm_vectors(psalms),
        "Full morphology baseline, psalm-broadcast: sp + all 9 atomic features, dim 77.",
    ):
        written.append("morph_full_all_psalm")

    return written


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus] = Corpus.load,
) -> None:
    """Generates every missing atomic-morphology dataset."""
    run_generator(__doc__, generate, argv, corpus_factory=corpus_factory)


if __name__ == "__main__":
    main()
