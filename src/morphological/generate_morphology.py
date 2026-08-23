"""Computes and writes the atomic-morphology family: 9 features, atomic and sp+feature."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from lexical.export import dataset_path, write_dataset
from morphological.atomic import (
    FeatureKey,
    atomic_psalm_vectors,
    atomic_vectors,
    full_morphology_psalm_vectors,
    full_morphology_vectors,
    sp_plus_feature_psalm_vectors,
    sp_plus_feature_vectors,
)
from morphological.corpus import Corpus, MorphologicalPsalm

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

_DATASET_TYPE = "morphological"


def _write_if_missing(
    output_root: Path,
    unit: str,
    construction: str,
    vectors: dict[int, np.ndarray],
    description: str,
) -> bool:
    if dataset_path(output_root, unit, construction, dataset_type=_DATASET_TYPE).exists():
        return False
    write_dataset(output_root, unit, construction, vectors, description, dataset_type=_DATASET_TYPE)
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
                f"computing morphological unit={unit} construction={construction}...",
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
                f"computing morphological unit={unit} construction={construction}...",
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

    print("computing morphological unit=morph_full construction=all...", file=sys.stderr)
    if _write_if_missing(
        output_root,
        "morph_full",
        "all",
        full_morphology_vectors(psalms),
        "Full morphology baseline: sp + all 9 atomic features concatenated, dim 77.",
    ):
        written.append("morph_full_all")

    print("computing morphological unit=morph_full construction=all_psalm...", file=sys.stderr)
    if _write_if_missing(
        output_root,
        "morph_full",
        "all_psalm",
        full_morphology_psalm_vectors(psalms),
        "Full morphology baseline, psalm-broadcast: sp + all 9 atomic features, dim 77.",
    ):
        written.append("morph_full_all_psalm")

    return written


def main() -> None:
    """Generates every missing atomic-morphology dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    output_root = parser.parse_args().output_root
    corpus = Corpus.load()
    psalms = corpus.psalms()
    written = generate(psalms, output_root)
    print(f"wrote {len(written)} dataset files", file=sys.stderr)


if __name__ == "__main__":
    main()
