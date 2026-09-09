"""Computes and writes clause_tab: BHSA hierarchical depth and its within-colon contour."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.cli import run_generator
from syntactic.clause_tab import (
    clause_tab_inventory_psalm_vectors,
    clause_tab_inventory_vectors,
    clause_tab_transition_psalm_vectors,
)
from syntactic.corpus import ClausePsalm, Corpus, clause_corpus
from syntactic.skeleton import generate_skeleton

_UNIT = "tab"
#: BHSA documents tab's theoretical reading as unsettled, so it is named descriptively.
_DESCRIPTION = "BHSA hierarchical clause-atom depth (capped at 10, majority colon assignment)"


def generate(psalms: list[ClausePsalm], output_root: Path) -> list[str]:
    """Writes every not-yet-written clause_tab construction, returns the names written."""
    return generate_skeleton(
        psalms,
        output_root,
        _UNIT,
        (
            ("inventory", clause_tab_inventory_vectors),
            ("inventory_psalm", clause_tab_inventory_psalm_vectors),
            ("transition_psalm", clause_tab_transition_psalm_vectors),
        ),
        _DESCRIPTION,
        level="clause",
    )


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus[ClausePsalm]] = clause_corpus,
) -> None:
    """Generates every missing clause_tab dataset."""
    run_generator(__doc__, generate, argv, corpus_factory=corpus_factory)


if __name__ == "__main__":
    main()
