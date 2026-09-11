"""Computes and writes the clause_typ family: 1gram, 1_2gram, and 1_2_3gram, both scales."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.cli import run_signature_generator
from core.supported_dataset import SupportedDataset, generate_supported_dataset
from syntactic import DATASET_TYPE
from syntactic.clause_support import MIN_EXTERNAL_SUPPORT_K_CLAUSE_ATOM_TYP
from syntactic.clause_typ_ngram import DENSE_BUILDERS, SPARSE_BUILDERS
from syntactic.corpus import ClausePsalm, Corpus, clause_corpus

_SUPPORT_FILE = "clause_atom_typ_external_support.csv"

DATASET: SupportedDataset[ClausePsalm] = SupportedDataset(
    unit="typ",
    domain=DATASET_TYPE,
    level="clause",
    dense=DENSE_BUILDERS,
    sparse=SPARSE_BUILDERS,
    description="Clause-atom type histogram (RARE-collapsed, k={k}, majority colon assignment)",
)


def generate(
    psalms: list[ClausePsalm],
    output_root: Path,
    external_counts: dict[str, int],
    k: int,
    *,
    max_workers: int | None = None,
) -> list[str]:
    """Writes every not-yet-written clause_typ construction, returns the names written."""
    return generate_supported_dataset(
        DATASET, psalms, output_root, external_counts, k, max_workers=max_workers
    )


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus[ClausePsalm]] = clause_corpus,
) -> None:
    """Generates every missing clause_typ dataset."""
    run_signature_generator(
        __doc__,
        generate,
        _SUPPORT_FILE,
        MIN_EXTERNAL_SUPPORT_K_CLAUSE_ATOM_TYP,
        argv,
        corpus_factory=corpus_factory,
    )


if __name__ == "__main__":
    main()
