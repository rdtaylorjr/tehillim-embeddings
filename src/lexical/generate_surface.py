"""Computes and writes surface-form datasets as Parquet: 19 weightings, each of three text tiers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from lexical.export import dataset_path, write_dataset
from lexical.frequency import icf_weights as compute_icf_weights
from lexical.frequency import total_token_count
from lexical.surface_corpus import SurfaceCorpus, SurfacePsalm
from lexical.surface_frequency import surface_token_frequencies
from lexical.surface_positional import surface_positional_icf_vectors
from lexical.surface_psalm_position import surface_psalm_positional_icf_vectors
from lexical.surface_psalm_recurrence import surface_psalm_spacing_profile_vectors
from lexical.surface_psalm_zoning import surface_psalm_position_mean_vectors
from lexical.surface_recurrence import surface_spacing_profile_vectors
from lexical.surface_vectorize import (
    surface_binary_presence_vectors,
    surface_icf_weighted_vectors,
    surface_log_count_vectors,
    surface_term_frequency_vectors,
    surface_tf_icf_vectors,
)
from lexical.surface_vocabulary import SurfaceTier, build_surface_vocabulary
from lexical.surface_zoning import surface_position_mean_vectors

_TIERS: tuple[SurfaceTier, ...] = ("consonantal", "vocalized", "cantillation")

_FULL_WEIGHTS = (
    "binary",
    "count",
    "log_count",
    "icf",
    "tf_icf",
    "icf_position2",
    "icf_position4",
    "icf_position8",
    "icf_spacing2",
    "icf_spacing4",
    "icf_spacing8",
    "icf_position_mean",
    "icf_position2_psalm",
    "icf_position4_psalm",
    "icf_position8_psalm",
    "icf_spacing2_psalm",
    "icf_spacing4_psalm",
    "icf_spacing8_psalm",
    "icf_position_mean_psalm",
)
_POSITIONAL_K = {"icf_position2": 2, "icf_position4": 4, "icf_position8": 8}
_SPACING_K = {"icf_spacing2": 2, "icf_spacing4": 4, "icf_spacing8": 8}
_PSALM_POSITIONAL_K = {"icf_position2_psalm": 2, "icf_position4_psalm": 4, "icf_position8_psalm": 8}
_PSALM_SPACING_K = {"icf_spacing2_psalm": 2, "icf_spacing4_psalm": 4, "icf_spacing8_psalm": 8}


def _vectors_for_weight(
    psalms: list[SurfacePsalm],
    vocabulary: tuple[str, ...],
    tier: SurfaceTier,
    weight: str,
    icf_weights: dict[str, float],
) -> dict[int, np.ndarray]:
    """Dispatches to the vector-building function matching `weight`."""
    if weight == "binary":
        return surface_binary_presence_vectors(psalms, vocabulary, tier)
    if weight == "count":
        return surface_term_frequency_vectors(psalms, vocabulary, tier)
    if weight == "log_count":
        return surface_log_count_vectors(psalms, vocabulary, tier)
    if weight == "icf":
        return surface_icf_weighted_vectors(psalms, vocabulary, tier, icf_weights)
    if weight == "tf_icf":
        return surface_tf_icf_vectors(psalms, vocabulary, tier, icf_weights)
    if weight in _POSITIONAL_K:
        return surface_positional_icf_vectors(
            psalms, vocabulary, tier, icf_weights, k=_POSITIONAL_K[weight]
        )
    if weight in _SPACING_K:
        return surface_spacing_profile_vectors(
            psalms, vocabulary, tier, icf_weights, k=_SPACING_K[weight]
        )
    if weight == "icf_position_mean":
        return surface_position_mean_vectors(psalms, vocabulary, tier, icf_weights)
    if weight in _PSALM_POSITIONAL_K:
        return surface_psalm_positional_icf_vectors(
            psalms, vocabulary, tier, icf_weights, k=_PSALM_POSITIONAL_K[weight]
        )
    if weight in _PSALM_SPACING_K:
        return surface_psalm_spacing_profile_vectors(
            psalms, vocabulary, tier, icf_weights, k=_PSALM_SPACING_K[weight]
        )
    if weight == "icf_position_mean_psalm":
        return surface_psalm_position_mean_vectors(psalms, vocabulary, tier, icf_weights)
    raise ValueError(f"unknown weight {weight!r}")


def generate_surface(
    psalms: list[SurfacePsalm],
    output_root: Path,
    icf_weights_by_tier: dict[SurfaceTier, dict[str, float]],
) -> list[str]:
    """Writes every not-yet-written (tier, weight) surface dataset, returns the names written."""
    written: list[str] = []
    for tier in _TIERS:
        vocabulary = build_surface_vocabulary(psalms, tier=tier)
        for weight in _FULL_WEIGHTS:
            if dataset_path(output_root, "word", weight, text=tier).exists():
                continue
            print(
                f"computing surface unit=word text={tier} construction={weight}...",
                file=sys.stderr,
            )
            vectors = _vectors_for_weight(
                psalms, vocabulary, tier, weight, icf_weights_by_tier[tier]
            )
            description = (
                f"Surface word-form vectors, {tier} text, construction={weight}, "
                f"dimension {len(vocabulary)}."
            )
            write_dataset(output_root, "word", weight, vectors, description, text=tier)
            written.append(f"word_{tier}_{weight}")
    return written


def main() -> None:
    """Generates every missing surface-form dataset: 19 weightings, each of three text tiers."""
    output_root = Path(__file__).resolve().parents[2]
    corpus = SurfaceCorpus.load()
    psalms = corpus.psalms()
    total_tokens = total_token_count(corpus.api)
    icf_weights_by_tier: dict[SurfaceTier, dict[str, float]] = {
        tier: compute_icf_weights(surface_token_frequencies(corpus.api, tier), total_tokens)
        for tier in _TIERS
    }
    written = generate_surface(psalms, output_root, icf_weights_by_tier)
    print(f"wrote {len(written)} dataset files", file=sys.stderr)


if __name__ == "__main__":
    main()
