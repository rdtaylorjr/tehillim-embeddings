"""Computes and writes surface-form datasets as Parquet: 19 weightings, each of three text tiers."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from core.cli import add_output_root_argument, report_generated
from core.export import path_to_write, write_vectors
from core.partition import BHSA_HALF_VERSE, SURFACE_TYPE, Partition, family
from core.spec import GeneratorSpec
from core.text import TextTier
from lexical import DOMAIN
from lexical.constructions import FULL_WEIGHTS, vectors_for_weight
from lexical.frequency import icf_weights as compute_icf_weights
from lexical.frequency import total_token_count
from lexical.surface_corpus import SurfaceCorpus, SurfacePsalm
from lexical.surface_frequency import surface_token_frequencies
from lexical.surface_vocabulary import build_surface_vocabulary, columns_for_tier

_TIERS: tuple[TextTier, ...] = ("consonantal", "vocalized", "cantillation")

SPEC = GeneratorSpec(
    module=__name__,
    partitions=tuple(
        p
        for tier in _TIERS
        for p in family(BHSA_HALF_VERSE, DOMAIN, FULL_WEIGHTS, type=SURFACE_TYPE, text=tier)
    ),
)


def generate_surface(
    psalms: list[SurfacePsalm],
    output_root: Path,
    icf_weights_by_tier: dict[TextTier, dict[str, float]],
) -> list[str]:
    """Writes every not-yet-written (tier, weight) surface dataset, returns the names written."""
    written: list[str] = []
    for tier in _TIERS:
        vocabulary = build_surface_vocabulary(psalms, tier=tier)
        columns = columns_for_tier(psalms, tier)
        for weight in FULL_WEIGHTS:
            partition = Partition(
                BHSA_HALF_VERSE, DOMAIN, type=SURFACE_TYPE, text=tier, construction=weight
            )
            path = path_to_write(output_root, partition)
            if path is None:
                continue
            vectors = vectors_for_weight(columns, vocabulary, weight, icf_weights_by_tier[tier])
            description = (
                f"Surface word-form vectors, {tier} text, construction={weight}, "
                f"dimension {len(vocabulary)}."
            )
            write_vectors(path, vectors, description)
            written.append(partition.identifier)
    return written


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], SurfaceCorpus] = SurfaceCorpus.load,
) -> None:
    """Generates every missing surface-form dataset: 19 weightings, each of three text tiers."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_output_root_argument(parser)
    args = parser.parse_args(argv)
    corpus = corpus_factory()
    psalms = corpus.psalms()
    total_tokens = total_token_count(corpus.api)
    icf_weights_by_tier: dict[TextTier, dict[str, float]] = {
        tier: compute_icf_weights(surface_token_frequencies(corpus.api, tier), total_tokens)
        for tier in _TIERS
    }
    written = generate_surface(psalms, args.output_root, icf_weights_by_tier)
    report_generated(written)


if __name__ == "__main__":
    main()
