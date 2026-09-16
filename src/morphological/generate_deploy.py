"""Computes and writes the psalm-scale grammatical deployment representation (Phase 4E)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.cli import run_generator
from core.export import path_to_write, write_vectors
from core.partition import BHSA_HALF_VERSE, Partition, family
from core.spec import GeneratorSpec
from morphological import DOMAIN, SUFFIX_FEATURE
from morphological.corpus import Corpus, MorphologicalPsalm
from morphological.deploy import suffix_deploy_vectors

CONSTRUCTION = "posmean"

SPEC = GeneratorSpec(
    module=__name__,
    partitions=family(BHSA_HALF_VERSE, DOMAIN, (CONSTRUCTION,), feature=SUFFIX_FEATURE),
)


def generate(psalms: list[MorphologicalPsalm], output_root: Path) -> list[str]:
    """Writes the morph_suffix posmean dataset if not already present, returns the names written."""
    partition = Partition(
        BHSA_HALF_VERSE, DOMAIN, feature=SUFFIX_FEATURE, construction=CONSTRUCTION
    )
    path = path_to_write(output_root, partition)
    if path is None:
        return []
    vectors = suffix_deploy_vectors(psalms)
    description = (
        f"Psalm-scale grammatical deployment [b;m] over the suffix vocabulary, "
        f"construction={CONSTRUCTION}."
    )
    write_vectors(path, vectors, description)
    return [partition.identifier]


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus] = Corpus.load,
) -> None:
    """Generates the morph_suffix posmean dataset if missing."""
    run_generator(__doc__, generate, argv, corpus_factory=corpus_factory)


if __name__ == "__main__":
    main()
