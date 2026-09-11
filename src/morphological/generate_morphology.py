"""Computes and writes the atomic-morphology family: 9 features, atomic, sp+feature, n-grams."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

from core.cli import run_generator
from core.dataset_family import Construction, generate_family
from core.ngram import concatenated_1_2_3gram_dim
from morphological import DATASET_TYPE
from morphological.atomic import (
    SPARSE_TRIGRAM_FEATURES,
    FeatureKey,
    feature_1_2_3gram_psalm_vectors,
    feature_1_2_3gram_vectors,
    feature_1_2gram_psalm_vectors,
    feature_1_2gram_vectors,
    feature_1gram_psalm_vectors,
    feature_1gram_vectors,
    feature_sparse_trigram_psalm_vectors,
    feature_sparse_trigram_vectors,
    full_morphology_psalm_vectors,
    full_morphology_vectors,
    sp_plus_feature_psalm_vectors,
    sp_plus_feature_vectors,
    vocabulary_for_feature,
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

#: Every construction of one feature, and the builder each is written from.
_BUILDERS = (
    ("1gram", feature_1gram_vectors),
    ("1gram_psalm", feature_1gram_psalm_vectors),
    ("sp_plus", sp_plus_feature_vectors),
    ("sp_plus_psalm", sp_plus_feature_psalm_vectors),
    ("1_2gram", feature_1_2gram_vectors),
    ("1_2gram_psalm", feature_1_2gram_psalm_vectors),
)

_TRIGRAMS = (
    ("1_2_3gram", feature_1_2_3gram_vectors, feature_sparse_trigram_vectors),
    ("1_2_3gram_psalm", feature_1_2_3gram_psalm_vectors, feature_sparse_trigram_psalm_vectors),
)


def _constructions(feature: FeatureKey) -> list[tuple[str, Construction[MorphologicalPsalm]]]:
    """Every construction of one feature, with the trigram stored sparsely where it must be."""
    describe = f"Atomic morphology histogram for {feature}, construction="
    family = [
        (
            name,
            Construction(build=partial(builder, feature=feature), description=f"{describe}{name}."),
        )
        for name, builder in _BUILDERS
    ]
    sparse = feature in SPARSE_TRIGRAM_FEATURES
    width = concatenated_1_2_3gram_dim(len(vocabulary_for_feature(feature)))
    family.extend(
        (
            name,
            Construction(
                build=partial(sparse_builder if sparse else dense_builder, feature=feature),
                description=f"{describe}{name}.",
                sparse_dim=width if sparse else None,
            ),
        )
        for name, dense_builder, sparse_builder in _TRIGRAMS
    )
    return family


def generate(psalms: list[MorphologicalPsalm], output_root: Path) -> list[str]:
    """Writes every not-yet-written atomic-morphology construction, returns the names written."""
    written: list[str] = []
    for feature in _FEATURES:
        written.extend(
            generate_family(
                psalms,
                output_root,
                _constructions(feature),
                unit=f"morph_{feature}",
                domain=DATASET_TYPE,
            )
        )
    written.extend(
        generate_family(
            psalms,
            output_root,
            [
                (
                    "all",
                    Construction(
                        build=full_morphology_vectors,
                        description=(
                            "Full morphology baseline: sp + all 9 atomic features concatenated, "
                            "dim 77."
                        ),
                    ),
                ),
                (
                    "all_psalm",
                    Construction(
                        build=full_morphology_psalm_vectors,
                        description=(
                            "Full morphology baseline, psalm-broadcast: sp + all 9 atomic "
                            "features, dim 77."
                        ),
                    ),
                ),
            ],
            unit="morph_full",
            domain=DATASET_TYPE,
        )
    )
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
