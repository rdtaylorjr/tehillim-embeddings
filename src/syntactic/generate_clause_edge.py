"""Computes and writes clause_mother: dependency span over BHSA mother links, both scales."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.cli import run_generator
from core.partition import BHSA_HALF_VERSE, family
from core.spec import GeneratorSpec
from syntactic import DOMAIN
from syntactic.clause_edge import (
    clause_edge_distance_psalm_vectors,
    clause_edge_distance_vectors,
)
from syntactic.corpus import ClausePsalm, Corpus, clause_corpus
from syntactic.skeleton import generate_skeleton

_FEATURE = "mother"
_DESCRIPTION = "Clause-atom dependency span (closed bins, majority half-verse assignment)"

_CONSTRUCTIONS = (
    ("edge_distance", clause_edge_distance_vectors),
    ("edge_distance_psalm", clause_edge_distance_psalm_vectors),
)

SPEC = GeneratorSpec(
    module=__name__,
    partitions=family(
        BHSA_HALF_VERSE,
        DOMAIN,
        tuple(name for name, _ in _CONSTRUCTIONS),
        level="clause",
        feature=_FEATURE,
    ),
)


def generate(psalms: list[ClausePsalm], output_root: Path) -> list[str]:
    """Writes both not-yet-written clause_mother constructions, returns the names written."""
    return generate_skeleton(
        psalms,
        output_root,
        _FEATURE,
        _CONSTRUCTIONS,
        _DESCRIPTION,
        level="clause",
    )


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus[ClausePsalm]] = clause_corpus,
) -> None:
    """Generates every missing clause_mother dataset."""
    run_generator(__doc__, generate, argv, corpus_factory=corpus_factory)


if __name__ == "__main__":
    main()
