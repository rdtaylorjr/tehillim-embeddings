"""Computes and writes the `[phrase_typ; phrase_function]` marginal baseline, both scales."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.cli import run_generator
from syntactic.corpus import Corpus, PhrasePsalm
from syntactic.marginal import typ_function_marginal_psalm_vectors, typ_function_marginal_vectors
from syntactic.skeleton import generate_skeleton

_UNIT = "marginal"
_DESCRIPTION = "Independent [typ; function] marginal histograms"


def generate(psalms: list[PhrasePsalm], output_root: Path) -> list[str]:
    """Writes both not-yet-written phrase_marginal constructions, returns the names written."""
    return generate_skeleton(
        psalms,
        output_root,
        _UNIT,
        (
            ("typ_function", typ_function_marginal_vectors),
            ("typ_function_psalm", typ_function_marginal_psalm_vectors),
        ),
        _DESCRIPTION,
    )


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus] = Corpus.load,
) -> None:
    """Generates every missing marginal dataset."""
    run_generator(__doc__, generate, argv, corpus_factory=corpus_factory)


if __name__ == "__main__":
    main()
