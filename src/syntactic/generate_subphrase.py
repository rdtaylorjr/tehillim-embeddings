"""Computes and writes the safe subphrase-relation skeleton: `rela=par` masked, both scales."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.cli import run_generator
from syntactic.corpus import Corpus, PhrasePsalm
from syntactic.skeleton import generate_skeleton
from syntactic.subphrase_vectorize import (
    subphrase_rela_1gram_psalm_vectors,
    subphrase_rela_1gram_vectors,
)

_UNIT = "subphrase_rela"
_DESCRIPTION = "Safe subphrase-relation skeleton (par masked)"


def generate(psalms: list[PhrasePsalm], output_root: Path) -> list[str]:
    """Writes both not-yet-written subphrase_rela constructions, returns the names written."""
    return generate_skeleton(
        psalms,
        output_root,
        _UNIT,
        (
            ("1gram", subphrase_rela_1gram_vectors),
            ("1gram_psalm", subphrase_rela_1gram_psalm_vectors),
        ),
        _DESCRIPTION,
    )


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus] = Corpus.load,
) -> None:
    """Generates every missing subphrase_rela dataset."""
    run_generator(__doc__, generate, argv, corpus_factory=corpus_factory)


if __name__ == "__main__":
    main()
