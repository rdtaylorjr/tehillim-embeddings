"""Computes and writes the phrase-complexity structural summary: half-verse and psalm-broadcast."""

from __future__ import annotations

from pathlib import Path

from syntax.complexity import phrase_complexity_psalm_vectors, phrase_complexity_vectors
from syntax.corpus import PhrasePsalm
from syntax.skeleton import generate_skeleton, run

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
    )


def main() -> None:
    """Generates every missing complexity dataset."""
    run(__doc__, generate)


if __name__ == "__main__":
    main()
