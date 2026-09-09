"""Computes and writes the safe phrase-relation skeleton: `rela=Para` masked, both scales."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.cli import run_generator
from syntactic.corpus import Corpus, PhrasePsalm
from syntactic.rela_vectorize import phrase_rela_1gram_psalm_vectors, phrase_rela_1gram_vectors
from syntactic.skeleton import generate_skeleton

_UNIT = "rela"
_DESCRIPTION = "Safe phrase-relation skeleton (Para masked)"


def generate(psalms: list[PhrasePsalm], output_root: Path) -> list[str]:
    """Writes both not-yet-written phrase_rela constructions, returns the names written."""
    return generate_skeleton(
        psalms,
        output_root,
        _UNIT,
        (("1gram", phrase_rela_1gram_vectors), ("1gram_psalm", phrase_rela_1gram_psalm_vectors)),
        _DESCRIPTION,
    )


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus] = Corpus.load,
) -> None:
    """Generates every missing phrase_rela dataset."""
    run_generator(__doc__, generate, argv, corpus_factory=corpus_factory)


if __name__ == "__main__":
    main()
