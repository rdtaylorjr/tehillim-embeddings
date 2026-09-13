"""Computes and writes the safe subphrase-relation skeleton: `rela=par` masked, both scales."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.cli import run_generator
from core.spec import GeneratorSpec, partition_dirs
from syntactic import DATASET_TYPE
from syntactic.corpus import Corpus, PhrasePsalm, phrase_corpus
from syntactic.skeleton import generate_skeleton
from syntactic.subphrase_vectorize import (
    subphrase_rela_1_2_3gram_psalm_vectors,
    subphrase_rela_1_2_3gram_vectors,
    subphrase_rela_1_2gram_psalm_vectors,
    subphrase_rela_1_2gram_vectors,
    subphrase_rela_1gram_psalm_vectors,
    subphrase_rela_1gram_vectors,
)

_UNIT = "subphrase_rela"
_DESCRIPTION = "Safe subphrase-relation skeleton (par masked)"

_CONSTRUCTIONS = (
    ("1gram", subphrase_rela_1gram_vectors),
    ("1gram_psalm", subphrase_rela_1gram_psalm_vectors),
    ("1_2gram", subphrase_rela_1_2gram_vectors),
    ("1_2gram_psalm", subphrase_rela_1_2gram_psalm_vectors),
    ("1_2_3gram", subphrase_rela_1_2_3gram_vectors),
    ("1_2_3gram_psalm", subphrase_rela_1_2_3gram_psalm_vectors),
)

SPEC = GeneratorSpec(
    module=__name__,
    partitions=partition_dirs(
        DATASET_TYPE, "feature", _UNIT, tuple(name for name, _ in _CONSTRUCTIONS), level="phrase"
    ),
)


def generate(psalms: list[PhrasePsalm], output_root: Path) -> list[str]:
    """Writes every not-yet-written subphrase_rela construction, returns the names written."""
    return generate_skeleton(
        psalms,
        output_root,
        _UNIT,
        _CONSTRUCTIONS,
        _DESCRIPTION,
        level="phrase",
    )


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus[PhrasePsalm]] = phrase_corpus,
) -> None:
    """Generates every missing subphrase_rela dataset."""
    run_generator(__doc__, generate, argv, corpus_factory=corpus_factory)


if __name__ == "__main__":
    main()
