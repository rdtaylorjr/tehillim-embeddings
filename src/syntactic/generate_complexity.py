"""Computes and writes the phrase-complexity structural summary: half-verse and psalm-broadcast."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.cli import run_generator
from syntactic.complexity import phrase_complexity_psalm_vectors, phrase_complexity_vectors
from syntactic.corpus import Corpus, PhrasePsalm, phrase_corpus
from syntactic.skeleton import generate_skeleton

_UNIT = "complexity"
_DESCRIPTION = (
    "Structural complexity [n_atoms; n_phrases; mean_words_per_atom; proportion_multi_atom]"
)


def generate(psalms: list[PhrasePsalm], output_root: Path) -> list[str]:
    """Writes both not-yet-written complexity constructions, returns the names written."""
    return generate_skeleton(
        psalms,
        output_root,
        _UNIT,
        (
            ("core", phrase_complexity_vectors),
            ("core_psalm", phrase_complexity_psalm_vectors),
        ),
        _DESCRIPTION,
        level="phrase",
    )


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus[PhrasePsalm]] = phrase_corpus,
) -> None:
    """Generates every missing complexity dataset."""
    run_generator(__doc__, generate, argv, corpus_factory=corpus_factory)


if __name__ == "__main__":
    main()
