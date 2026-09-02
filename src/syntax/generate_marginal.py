"""Computes and writes the `[phrase_typ; phrase_function]` marginal baseline, both scales."""

from __future__ import annotations

from pathlib import Path

from syntax.corpus import PhrasePsalm
from syntax.marginal import typ_function_marginal_psalm_vectors, typ_function_marginal_vectors
from syntax.skeleton import generate_skeleton, run

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


def main() -> None:
    """Generates every missing marginal dataset."""
    run(__doc__, generate)


if __name__ == "__main__":
    main()
