"""Computes and writes the clause_rela family: the safe relation inventory, both scales."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.cli import run_signature_generator
from syntactic.clause_generator import ClauseFamily, generate_family
from syntactic.clause_rela_vectorize import (
    clause_rela_1_2_3gram_psalm_vectors,
    clause_rela_1_2_3gram_vectors,
    clause_rela_1_2gram_psalm_vectors,
    clause_rela_1_2gram_vectors,
    clause_rela_1gram_psalm_vectors,
    clause_rela_1gram_vectors,
)
from syntactic.clause_support import MIN_EXTERNAL_SUPPORT_K_CLAUSE_RELA
from syntactic.corpus import ClausePsalm, Corpus, clause_corpus

FAMILY = ClauseFamily(
    unit="rela",
    description="Safe clause-relation inventory (Resu and ReVo removed)",
    dense={
        "1gram": clause_rela_1gram_vectors,
        "1gram_psalm": clause_rela_1gram_psalm_vectors,
        "1_2gram": clause_rela_1_2gram_vectors,
        "1_2gram_psalm": clause_rela_1_2gram_psalm_vectors,
        "1_2_3gram": clause_rela_1_2_3gram_vectors,
        "1_2_3gram_psalm": clause_rela_1_2_3gram_psalm_vectors,
    },
    sparse={},
)


def generate(
    psalms: list[ClausePsalm], output_root: Path, external_counts: dict[str, int], k: int
) -> list[str]:
    """Writes every not-yet-written clause_rela construction, returns the names written."""
    return generate_family(FAMILY, psalms, output_root, external_counts, k)


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus[ClausePsalm]] = clause_corpus,
) -> None:
    """Generates every missing clause_rela dataset."""
    run_signature_generator(
        __doc__,
        generate,
        "clause_rela_external_support.csv",
        MIN_EXTERNAL_SUPPORT_K_CLAUSE_RELA,
        argv,
        corpus_factory=corpus_factory,
    )


if __name__ == "__main__":
    main()
