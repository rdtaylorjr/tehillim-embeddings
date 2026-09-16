"""Computes and writes the safe phrase-relation skeleton: `rela=Para` masked, both scales."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.cli import run_generator
from core.partition import BHSA_HALF_VERSE, family
from core.spec import GeneratorSpec
from syntactic import DOMAIN
from syntactic.corpus import Corpus, PhrasePsalm, phrase_corpus
from syntactic.rela_vectorize import (
    phrase_rela_1_2_3gram_psalm_vectors,
    phrase_rela_1_2_3gram_vectors,
    phrase_rela_1_2gram_psalm_vectors,
    phrase_rela_1_2gram_vectors,
    phrase_rela_1gram_psalm_vectors,
    phrase_rela_1gram_vectors,
)
from syntactic.skeleton import generate_skeleton

_FEATURE = "rela"
_DESCRIPTION = "Safe phrase-relation skeleton (Para masked)"

_CONSTRUCTIONS = (
    ("1gram", phrase_rela_1gram_vectors),
    ("1gram_psalm", phrase_rela_1gram_psalm_vectors),
    ("1_2gram", phrase_rela_1_2gram_vectors),
    ("1_2gram_psalm", phrase_rela_1_2gram_psalm_vectors),
    ("1_2_3gram", phrase_rela_1_2_3gram_vectors),
    ("1_2_3gram_psalm", phrase_rela_1_2_3gram_psalm_vectors),
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
    """Writes every not-yet-written phrase_rela construction, returns the names written."""
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
    """Generates every missing phrase_rela dataset."""
    run_generator(__doc__, generate, argv, corpus_factory=corpus_factory)


if __name__ == "__main__":
    main()
