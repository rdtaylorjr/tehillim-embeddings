"""Computes and writes the phrase_signature family: inventory, 1_2gram, and 1_2_3gram."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core.cli import run_signature_generator
from core.export import path_to_write, write_sparse_vectors, write_vectors
from core.ngram import concatenated_1_2_3gram_dim
from core.parallel import map_constructions
from core.support import build_signature_vocabulary
from syntax import DATASET_TYPE, SIGNATURE_UNIT
from syntax.corpus import Corpus, PhrasePsalm
from syntax.signature_support import MIN_EXTERNAL_SUPPORT_K
from syntax.signature_vectorize import DENSE_BUILDERS, SPARSE_BUILDERS


@dataclass(frozen=True, slots=True)
class _Context:
    """Everything one construction needs, pickled once per worker rather than once per build."""

    psalms: tuple[PhrasePsalm, ...]
    output_root: Path
    vocabulary: tuple[str, ...]
    external_counts: dict[str, int]
    k: int


def write_construction(context: _Context, construction: str) -> str | None:
    """Writes one construction's dataset, or returns None when it is already written."""
    path = path_to_write(
        context.output_root,
        SIGNATURE_UNIT,
        construction,
        domain=DATASET_TYPE,
        unit_key="feature",
        level="phrase",
    )
    if path is None:
        return None
    args = (list(context.psalms), context.vocabulary, context.external_counts, context.k)
    description = (
        f"Phrase-signature histogram (RARE-collapsed, k={context.k}), construction={construction}."
    )
    dense_builder = DENSE_BUILDERS.get(construction)
    if dense_builder is not None:
        write_vectors(path, dense_builder(*args), description)
    else:
        sparse_dim = concatenated_1_2_3gram_dim(len(context.vocabulary))
        write_sparse_vectors(path, SPARSE_BUILDERS[construction](*args), sparse_dim, description)
    return f"{SIGNATURE_UNIT}_{construction}"


def generate(
    psalms: list[PhrasePsalm],
    output_root: Path,
    external_counts: dict[str, int],
    k: int,
    *,
    max_workers: int | None = None,
) -> list[str]:
    """Writes every not-yet-written phrase_signature construction, returns the names written."""
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
    corpus_factory: Callable[[], Corpus] = Corpus.load,
) -> None:
    """Generates every missing phrase_signature dataset."""
    run_signature_generator(
        __doc__,
        generate,
        "phrase_signature_external_support.csv",
        MIN_EXTERNAL_SUPPORT_K,
        argv,
        corpus_factory=corpus_factory,
    )


if __name__ == "__main__":
    main()
