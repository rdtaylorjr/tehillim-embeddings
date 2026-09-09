"""Computes and writes the phrase-determination (det) skeleton: half-verse and psalm-broadcast."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.cli import run_generator
from syntactic.corpus import Corpus, PhrasePsalm, phrase_corpus
from syntactic.det_vectorize import phrase_det_1gram_psalm_vectors, phrase_det_1gram_vectors
from syntactic.skeleton import generate_skeleton

_UNIT = "det"
_DESCRIPTION = "Phrase-determination skeleton"


def generate(psalms: list[PhrasePsalm], output_root: Path) -> list[str]:
    """Writes both not-yet-written phrase_det constructions, returns the names written."""
    return generate_skeleton(
        psalms,
        output_root,
        _UNIT,
        (("1gram", phrase_det_1gram_vectors), ("1gram_psalm", phrase_det_1gram_psalm_vectors)),
        _DESCRIPTION,
        level="phrase",
    )


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus[PhrasePsalm]] = phrase_corpus,
) -> None:
    """Generates every missing phrase_det dataset."""
    run_generator(__doc__, generate, argv, corpus_factory=corpus_factory)


if __name__ == "__main__":
    main()
