"""Computes and writes the clause_rela family: the safe relation inventory, both scales."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.cli import run_signature_generator
from core.partition import BHSA_HALF_VERSE, family
from core.spec import GeneratorSpec
from syntactic import DOMAIN
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
    feature="rela",
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

_SUPPORT_FILE = "clause_rela_external_support.csv"

SPEC = GeneratorSpec(
    module=__name__,
    partitions=family(
        BHSA_HALF_VERSE,
        DOMAIN,
        (*FAMILY.dense, *FAMILY.sparse),
        level="clause",
        feature=FAMILY.feature,
    ),
    support=(_SUPPORT_FILE,),
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
        _SUPPORT_FILE,
        MIN_EXTERNAL_SUPPORT_K_CLAUSE_RELA,
        argv,
        corpus_factory=corpus_factory,
    )


if __name__ == "__main__":
    main()
