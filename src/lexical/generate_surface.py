"""Computes and writes surface-form datasets as Parquet: 19 weightings, each of three text tiers."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from core.export import dataset_path, write_dataset
from lexical.constructions import FULL_WEIGHTS, vectors_for_weight
from lexical.frequency import icf_weights as compute_icf_weights
from lexical.frequency import total_token_count
from lexical.surface_corpus import SurfaceCorpus, SurfacePsalm
from lexical.surface_frequency import surface_token_frequencies
from lexical.surface_vocabulary import SurfaceTier, build_surface_vocabulary, columns_for_tier

_TIERS: tuple[SurfaceTier, ...] = ("consonantal", "vocalized", "cantillation")


def generate_surface(
    psalms: list[SurfacePsalm],
    output_root: Path,
    icf_weights_by_tier: dict[SurfaceTier, dict[str, float]],
) -> list[str]:
    """Writes every not-yet-written (tier, weight) surface dataset, returns the names written."""
    written: list[str] = []
    for tier in _TIERS:
        vocabulary = build_surface_vocabulary(psalms, tier=tier)
        columns = columns_for_tier(psalms, tier)
        for weight in FULL_WEIGHTS:
            if dataset_path(output_root, "word", weight, text=tier, unit_key="unit").exists():
                continue
            print(
                f"computing surface unit=word text={tier} construction={weight}...",
                file=sys.stderr,
            )
            vectors = vectors_for_weight(columns, vocabulary, weight, icf_weights_by_tier[tier])
            description = (
                f"Surface word-form vectors, {tier} text, construction={weight}, "
                f"dimension {len(vocabulary)}."
            )
            write_dataset(
                output_root, "word", weight, vectors, description, text=tier, unit_key="unit"
            )
            written.append(f"word_{tier}_{weight}")
    return written


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], SurfaceCorpus] = SurfaceCorpus.load,
) -> None:
    """Generates every missing surface-form dataset: 19 weightings, each of three text tiers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    corpus = corpus_factory()
    psalms = corpus.psalms()
    total_tokens = total_token_count(corpus.api)
    icf_weights_by_tier: dict[SurfaceTier, dict[str, float]] = {
        tier: compute_icf_weights(surface_token_frequencies(corpus.api, tier), total_tokens)
        for tier in _TIERS
    }
    written = generate_surface(psalms, args.output_root, icf_weights_by_tier)
    print(f"wrote {len(written)} dataset files", file=sys.stderr)


if __name__ == "__main__":
    main()
