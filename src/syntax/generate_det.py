"""Computes and writes the phrase-determination (det) skeleton: half-verse and psalm-broadcast."""

from __future__ import annotations

from pathlib import Path

from syntax.corpus import PhrasePsalm
from syntax.det_vectorize import phrase_det_1gram_psalm_vectors, phrase_det_1gram_vectors
from syntax.skeleton import generate_skeleton, run

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
    )


def main() -> None:
    """Generates every missing phrase_det dataset."""
    run(__doc__, generate)


if __name__ == "__main__":
    main()
