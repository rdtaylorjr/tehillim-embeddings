"""Computes and writes the clause-kind skeleton: the coarse verbal/nominal split, both scales."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.cli import run_generator
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

_UNIT = "kind"
_DESCRIPTION = "Clause-kind inventory (VC/NC/WP), majority colon assignment"


def generate(psalms: list[ClausePsalm], output_root: Path) -> list[str]:
    """Writes every not-yet-written clause_kind constructions, returns the names written."""
    return generate_skeleton(
        psalms,
        output_root,
        _UNIT,
        (
            ("1gram", clause_kind_1gram_vectors),
            ("1gram_psalm", clause_kind_1gram_psalm_vectors),
            ("1_2gram", clause_kind_1_2gram_vectors),
            ("1_2gram_psalm", clause_kind_1_2gram_psalm_vectors),
            ("1_2_3gram", clause_kind_1_2_3gram_vectors),
            ("1_2_3gram_psalm", clause_kind_1_2_3gram_psalm_vectors),
        ),
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
