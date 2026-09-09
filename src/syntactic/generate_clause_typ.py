"""Computes and writes the clause_typ family: 1gram, 1_2gram, and 1_2_3gram, both scales."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core.cli import run_signature_generator
from core.export import path_to_write, write_sparse_vectors, write_vectors
from core.ngram import concatenated_1_2_3gram_dim
from core.parallel import map_constructions
from core.support import build_signature_vocabulary
from syntactic import DATASET_TYPE
from syntactic.clause_support import MIN_EXTERNAL_SUPPORT_K_CLAUSE_ATOM_TYP
from syntactic.clause_typ_ngram import DENSE_BUILDERS, SPARSE_BUILDERS
from syntactic.corpus import ClausePsalm, Corpus, clause_corpus

_UNIT = "typ"
_SUPPORT_FILE = "clause_atom_typ_external_support.csv"


@dataclass(frozen=True, slots=True)
class _Context:
    """Everything one construction needs, pickled once per worker rather than once per build."""

    psalms: tuple[ClausePsalm, ...]
    output_root: Path
    vocabulary: tuple[str, ...]
    external_counts: dict[str, int]
    k: int


def write_construction(context: _Context, construction: str) -> str | None:
    """Writes one construction's dataset, or returns None when it is already written."""
    path = path_to_write(
        context.output_root,
        _UNIT,
        construction,
        domain=DATASET_TYPE,
        unit_key="feature",
        level="clause",
    )
    if path is None:
        return None
    args = (list(context.psalms), context.vocabulary, context.external_counts, context.k)
    description = (
        f"Clause-atom type histogram (RARE-collapsed, k={context.k}, majority colon "
        f"assignment), construction={construction}."
    )
    dense_builder = DENSE_BUILDERS.get(construction)
    if dense_builder is not None:
        write_vectors(path, dense_builder(*args), description)
    else:
        sparse_dim = concatenated_1_2_3gram_dim(len(context.vocabulary))
        write_sparse_vectors(path, SPARSE_BUILDERS[construction](*args), sparse_dim, description)
    return f"{_UNIT}_{construction}"


def generate(
    psalms: list[ClausePsalm],
    output_root: Path,
    external_counts: dict[str, int],
    k: int,
    *,
    max_workers: int | None = None,
) -> list[str]:
    """Writes every not-yet-written clause_typ construction, returns the names written."""
    context = _Context(
        psalms=tuple(psalms),
        output_root=output_root,
        vocabulary=build_signature_vocabulary(external_counts, k),
        external_counts=external_counts,
        k=k,
    )
    constructions = (*DENSE_BUILDERS, *SPARSE_BUILDERS)
    return map_constructions(write_construction, context, constructions, max_workers=max_workers)


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
