"""Computes and writes the `[phrase_typ; phrase_function]` marginal baseline, both scales."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.cli import run_generator
from core.partition import BHSA_HALF_VERSE, family
from core.spec import GeneratorSpec
from syntactic import DOMAIN
from syntactic.corpus import Corpus, PhrasePsalm, phrase_corpus
from syntactic.marginal import typ_function_marginal_psalm_vectors, typ_function_marginal_vectors
from syntactic.skeleton import generate_skeleton

_FEATURE = "marginal"
_DESCRIPTION = "Independent [typ; function] marginal histograms"

_CONSTRUCTIONS = (
    ("typ_function", typ_function_marginal_vectors),
    ("typ_function_psalm", typ_function_marginal_psalm_vectors),
)

SPEC = GeneratorSpec(
    module=__name__,
    partitions=family(
        BHSA_HALF_VERSE,
        DOMAIN,
        tuple(name for name, _ in _CONSTRUCTIONS),
        level="phrase",
        feature=_FEATURE,
    ),
)


def generate(psalms: list[PhrasePsalm], output_root: Path) -> list[str]:
    """Writes both not-yet-written phrase_marginal constructions, returns the names written."""
    return generate_skeleton(
        psalms,
        output_root,
        _FEATURE,
        _CONSTRUCTIONS,
        _DESCRIPTION,
        level="phrase",
    )


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus[PhrasePsalm]] = phrase_corpus,
) -> None:
    """Generates every missing marginal dataset."""
    run_generator(__doc__, generate, argv, corpus_factory=corpus_factory)


if __name__ == "__main__":
    main()
