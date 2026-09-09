"""Computes and writes morph_atomic (dim 66) and the morph_signature family of constructions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core.cli import run_signature_generator
from core.export import path_to_write, write_sparse_vectors, write_vectors
from core.ngram import concatenated_1_2_3gram_dim
from core.parallel import map_constructions
from core.support import build_signature_vocabulary
from morphological import ATOMIC_UNIT, DATASET_TYPE, SIGNATURE_UNIT
from morphological.corpus import Corpus, MorphologicalPsalm
from morphological.signature_support import MIN_EXTERNAL_SUPPORT_K
from morphological.signature_vectorize import (
    DENSE_BUILDERS,
    SPARSE_BUILDERS,
    morph_atomic_psalm_vectors,
    morph_atomic_vectors,
)


def _signature_description(k: int, construction: str) -> str:
    """The metadata sentence shared by every morph_signature construction."""
    return f"Grammatical-signature histogram (RARE-collapsed, k={k}), construction={construction}."


#: The atomic baseline is written alongside the signatures, keyed the same way for one dispatch.
ATOMIC_BUILDERS = {"core": morph_atomic_vectors, "core_psalm": morph_atomic_psalm_vectors}


@dataclass(frozen=True, slots=True)
class _Context:
    """Everything one construction needs, pickled once per worker rather than once per build."""

    psalms: tuple[MorphologicalPsalm, ...]
    output_root: Path
    vocabulary: tuple[str, ...]
    external_counts: dict[str, int]
    k: int


def write_construction(context: _Context, construction: str) -> str | None:
    """Writes one construction's dataset, or returns None when it is already written."""
    psalms = list(context.psalms)
    atomic_builder = ATOMIC_BUILDERS.get(construction)
    unit = ATOMIC_UNIT if atomic_builder is not None else SIGNATURE_UNIT
    path = path_to_write(
        context.output_root, unit, construction, domain=DATASET_TYPE, unit_key="feature"
    )
    if path is None:
        return None
    if atomic_builder is not None:
        description = (
            f"Atomic morphology baseline [sp;gn;nu;ps;st;vs;vt], construction={construction}."
        )
        write_vectors(path, atomic_builder(psalms), description)
        return f"morph_atomic_{construction}"

    args = (psalms, context.vocabulary, context.external_counts, context.k)
    description = _signature_description(context.k, construction)
    dense_builder = DENSE_BUILDERS.get(construction)
    if dense_builder is not None:
        write_vectors(path, dense_builder(*args), description)
    else:
        combined_dim = concatenated_1_2_3gram_dim(len(context.vocabulary))
        write_sparse_vectors(path, SPARSE_BUILDERS[construction](*args), combined_dim, description)
    return f"morph_signature_{construction}"


def generate(
    psalms: list[MorphologicalPsalm],
    output_root: Path,
    external_counts: dict[str, int],
    k: int,
    *,
    max_workers: int | None = None,
) -> list[str]:
    """Writes every not-yet-written morph_atomic/morph_signature construction, returns names."""
    context = _Context(
        psalms=tuple(psalms),
        output_root=output_root,
        vocabulary=build_signature_vocabulary(external_counts, k),
        external_counts=external_counts,
        k=k,
    )
    constructions = (*ATOMIC_BUILDERS, *DENSE_BUILDERS, *SPARSE_BUILDERS)
    return map_constructions(write_construction, context, constructions, max_workers=max_workers)


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus] = Corpus.load,
) -> None:
    """Generates every missing morph_atomic/morph_signature dataset."""
    run_signature_generator(
        __doc__,
        generate,
        "morph_signature_external_support.csv",
        MIN_EXTERNAL_SUPPORT_K,
        argv,
        corpus_factory=corpus_factory,
    )


if __name__ == "__main__":
    main()
