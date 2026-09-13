"""Computes and writes clause_tree: dependency-forest shape, psalm scale only."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.cli import run_generator
from core.spec import GeneratorSpec, partition_dirs
from syntactic import DATASET_TYPE
from syntactic.clause_tree import clause_tree_summary_psalm_vectors
from syntactic.corpus import ClausePsalm, Corpus, clause_corpus
from syntactic.skeleton import generate_skeleton

_UNIT = "tree"
#: Half-verse level is omitted deliberately: 65 percent of half-verses hold under two clause atoms.
_DESCRIPTION = "Clause-atom dependency-forest shape, psalm scale"

_CONSTRUCTIONS = (("summary_psalm", clause_tree_summary_psalm_vectors),)

SPEC = GeneratorSpec(
    module=__name__,
    partitions=partition_dirs(
        DATASET_TYPE, "feature", _UNIT, tuple(name for name, _ in _CONSTRUCTIONS), level="clause"
    ),
)


def generate(psalms: list[ClausePsalm], output_root: Path) -> list[str]:
    """Writes the not-yet-written clause_tree construction, returns the names written."""
    return generate_skeleton(
        psalms,
        output_root,
        _UNIT,
        _CONSTRUCTIONS,
        _DESCRIPTION,
        level="clause",
    )


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus[ClausePsalm]] = clause_corpus,
) -> None:
    """Generates every missing clause_tree dataset."""
    run_generator(__doc__, generate, argv, corpus_factory=corpus_factory)


if __name__ == "__main__":
    main()
