"""Computes and writes the safe phrase-relation skeleton: `rela=Para` masked, both scales."""

from __future__ import annotations

from pathlib import Path

from syntax.corpus import PhrasePsalm
from syntax.rela_vectorize import phrase_rela_1gram_psalm_vectors, phrase_rela_1gram_vectors
from syntax.skeleton import generate_skeleton, run

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


def main() -> None:
    """Generates every missing phrase_rela dataset."""
    run(__doc__, generate)


if __name__ == "__main__":
    main()
