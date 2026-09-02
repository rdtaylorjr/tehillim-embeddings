"""Computes and writes the safe subphrase-relation skeleton: `rela=par` masked, both scales."""

from __future__ import annotations

from pathlib import Path

from syntax.corpus import PhrasePsalm
from syntax.skeleton import generate_skeleton, run
from syntax.subphrase_vectorize import (
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


def main() -> None:
    """Generates every missing subphrase_rela dataset."""
    run(__doc__, generate)


if __name__ == "__main__":
    main()
