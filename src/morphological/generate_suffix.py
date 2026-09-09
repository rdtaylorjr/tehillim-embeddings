"""Computes and writes the morph_suffix family: suffix inventory, and host-signature-plus-suffix."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.cli import run_signature_generator
from core.export import path_to_write, write_vectors
from core.parallel import map_constructions
from core.support import build_signature_vocabulary
from morphological import DATASET_TYPE, SUFFIX_UNIT
from morphological.corpus import Corpus, MorphologicalPsalm
from morphological.signature_support import MIN_EXTERNAL_SUPPORT_K
from morphological.suffix import (
    host_plus_suffix_psalm_vectors,
    host_plus_suffix_vectors,
    suffix_inventory_psalm_vectors,
    suffix_inventory_vectors,
)

#: Keyed by construction so a worker resolves its builder by name, which a lambda cannot pickle.
_BUILDERS = ("inventory", "inventory_psalm", "host_plus_suffix", "host_plus_suffix_psalm")


@dataclass(frozen=True, slots=True)
class _Context:
    """Everything one construction needs, pickled once per worker rather than once per build."""

    psalms: tuple[MorphologicalPsalm, ...]
    output_root: Path
    signature_vocabulary: tuple[str, ...]
    external_counts: dict[str, int]
    k: int


def _build(context: _Context, construction: str) -> dict[int, np.ndarray]:
    """One construction's vectors, resolved by name inside the worker that will write them."""
    psalms = list(context.psalms)
    if construction == "inventory":
        return suffix_inventory_vectors(psalms)
    if construction == "inventory_psalm":
        return suffix_inventory_psalm_vectors(psalms)
    args = (psalms, context.signature_vocabulary, context.external_counts, context.k)
    if construction == "host_plus_suffix":
        return host_plus_suffix_vectors(*args)
    return host_plus_suffix_psalm_vectors(*args)


def write_construction(context: _Context, construction: str) -> str | None:
    """Writes one construction's dataset, or returns None when it is already written."""
    path = path_to_write(
        context.output_root, SUFFIX_UNIT, construction, domain=DATASET_TYPE, unit_key="feature"
    )
    if path is None:
        return None
    description = f"Pronominal-suffix representation, construction={construction}."
    write_vectors(path, _build(context, construction), description)
    return f"morph_suffix_{construction}"


def generate(
    psalms: list[MorphologicalPsalm],
    output_root: Path,
    external_counts: dict[str, int],
    k: int,
    *,
    max_workers: int | None = None,
) -> list[str]:
    """Writes every not-yet-written morph_suffix construction, returns the names written."""
    context = _Context(
        psalms=tuple(psalms),
        output_root=output_root,
        signature_vocabulary=build_signature_vocabulary(external_counts, k),
        external_counts=external_counts,
        k=k,
    )
    return map_constructions(write_construction, context, _BUILDERS, max_workers=max_workers)


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus] = Corpus.load,
) -> None:
    """Generates every missing morph_suffix dataset."""
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
