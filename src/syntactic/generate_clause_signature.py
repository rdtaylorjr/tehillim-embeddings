"""Computes and writes the clause_signature family: inventory, 1_2gram, and 1_2_3gram."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.cli import run_signature_generator
from core.spec import GeneratorSpec, partition_dirs
from syntactic import DATASET_TYPE
from syntactic.clause_generator import ClauseFamily, generate_family
from syntactic.clause_rela_vectorize import (
    SIGNATURE_DENSE_BUILDERS,
    SIGNATURE_SPARSE_BUILDERS,
)
from syntactic.clause_support import MIN_EXTERNAL_SUPPORT_K_CLAUSE_SIGNATURE
from syntactic.corpus import ClausePsalm, Corpus, clause_corpus

FAMILY = ClauseFamily(
    unit="signature",
    description="Joint clause typ:rela signature (Resu and ReVo removed)",
    dense=SIGNATURE_DENSE_BUILDERS,
    sparse=SIGNATURE_SPARSE_BUILDERS,
)

_SUPPORT_FILE = "clause_signature_external_support.csv"

SPEC = GeneratorSpec(
    module=__name__,
    partitions=partition_dirs(
        DATASET_TYPE, "feature", FAMILY.unit, (*FAMILY.dense, *FAMILY.sparse), level="clause"
    ),
    support=(_SUPPORT_FILE,),
)


def generate(
    psalms: list[ClausePsalm], output_root: Path, external_counts: dict[str, int], k: int
) -> list[str]:
    """Writes every not-yet-written clause_signature construction, returns the names written."""
    return generate_family(FAMILY, psalms, output_root, external_counts, k)


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus[ClausePsalm]] = clause_corpus,
) -> None:
    """Generates every missing clause_signature dataset."""
    run_signature_generator(
        __doc__,
        generate,
        _SUPPORT_FILE,
        MIN_EXTERNAL_SUPPORT_K_CLAUSE_SIGNATURE,
        argv,
        corpus_factory=corpus_factory,
    )


if __name__ == "__main__":
    main()
