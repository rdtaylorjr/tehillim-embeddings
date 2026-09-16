"""Computes and writes the clause-kind skeleton: the coarse verbal/nominal split, both scales."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.cli import run_generator
from core.partition import BHSA_HALF_VERSE, family
from core.spec import GeneratorSpec
from syntactic import DOMAIN
from syntactic.clause_kind import (
    clause_kind_1_2_3gram_psalm_vectors,
    clause_kind_1_2_3gram_vectors,
    clause_kind_1_2gram_psalm_vectors,
    clause_kind_1_2gram_vectors,
    clause_kind_1gram_psalm_vectors,
    clause_kind_1gram_vectors,
)
from syntactic.corpus import ClausePsalm, Corpus, clause_corpus
from syntactic.skeleton import generate_skeleton

_FEATURE = "kind"
_DESCRIPTION = "Clause-kind inventory (VC/NC/WP), majority half-verse assignment"

_CONSTRUCTIONS = (
    ("1gram", clause_kind_1gram_vectors),
    ("1gram_psalm", clause_kind_1gram_psalm_vectors),
    ("1_2gram", clause_kind_1_2gram_vectors),
    ("1_2gram_psalm", clause_kind_1_2gram_psalm_vectors),
    ("1_2_3gram", clause_kind_1_2_3gram_vectors),
    ("1_2_3gram_psalm", clause_kind_1_2_3gram_psalm_vectors),
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
    """Writes every not-yet-written clause_kind constructions, returns the names written."""
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
    """Generates every missing clause_kind dataset."""
    run_generator(__doc__, generate, argv, corpus_factory=corpus_factory)


if __name__ == "__main__":
    main()
