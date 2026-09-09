"""Computes and writes clause_marginal: the independent-marginals control for H6.5."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from core.cli import add_config_root_argument, add_output_root_argument, report_generated
from core.export import path_to_write, write_vectors
from core.support import build_signature_vocabulary, load_external_signature_counts
from syntactic import DATASET_TYPE
from syntactic.clause_marginal import (
    MarginalSide,
    clause_marginal_psalm_vectors,
    clause_marginal_vectors,
)
from syntactic.clause_support import (
    MIN_EXTERNAL_SUPPORT_K_CLAUSE_RELA,
    MIN_EXTERNAL_SUPPORT_K_CLAUSE_TYP,
)
from syntactic.corpus import ClausePsalm, Corpus, clause_corpus

_UNIT = "marginal"
_BUILDERS = {"typ_rela": clause_marginal_vectors, "typ_rela_psalm": clause_marginal_psalm_vectors}


def _side(config_root: Path, filename: str, k: int) -> MarginalSide:
    """Loads one side's frozen support table and the vocabulary it yields."""
    counts = load_external_signature_counts(config_root / filename)
    return MarginalSide(build_signature_vocabulary(counts, k), counts, k)


def generate(
    psalms: list[ClausePsalm], output_root: Path, typ: MarginalSide, rela: MarginalSide
) -> list[str]:
    """Writes every not-yet-written clause_marginal construction, returns the names written."""
    written: list[str] = []
    for construction, builder in _BUILDERS.items():
        path = path_to_write(
            output_root,
            _UNIT,
            construction,
            domain=DATASET_TYPE,
            unit_key="feature",
            level="clause",
        )
        if path is None:
            continue
        description = (
            f"Independent clause typ and rela marginals (Resu and ReVo removed, "
            f"typ k={typ.k}, rela k={rela.k}), construction={construction}."
        )
        write_vectors(path, builder(psalms, typ, rela), description)
        written.append(f"{_UNIT}_{construction}")
    return written


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus[ClausePsalm]] = clause_corpus,
) -> None:
    """Generates every missing clause_marginal dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_output_root_argument(parser)
    add_config_root_argument(parser)
    args = parser.parse_args(argv)
    typ = _side(
        args.config_root, "clause_typ_external_support.csv", MIN_EXTERNAL_SUPPORT_K_CLAUSE_TYP
    )
    rela = _side(
        args.config_root, "clause_rela_external_support.csv", MIN_EXTERNAL_SUPPORT_K_CLAUSE_RELA
    )
    report_generated(generate(corpus_factory().psalms(), args.output_root, typ, rela))


if __name__ == "__main__":
    main()
