"""Shared driver for clause generators whose vocabulary comes from a frozen support table."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.export import path_to_write, write_sparse_vectors, write_vectors
from core.ngram import concatenated_1_2_3gram_dim
from core.parallel import map_constructions
from core.support import build_signature_vocabulary
from syntactic import DATASET_TYPE
from syntactic.corpus import ClausePsalm

type DenseBuilder = Callable[
    [list[ClausePsalm], tuple[str, ...], dict[str, int], int], dict[int, np.ndarray]
]
type SparseBuilder = Callable[
    [list[ClausePsalm], tuple[str, ...], dict[str, int], int],
    dict[int, tuple[np.ndarray, np.ndarray]],
]

__all__ = ["ClauseFamily", "generate_family"]


@dataclass(frozen=True, slots=True)
class ClauseFamily:
    """One clause feature's unit name, prose, and the builders each construction dispatches to."""

    unit: str
    description: str
    dense: Mapping[str, DenseBuilder]
    sparse: Mapping[str, SparseBuilder]


@dataclass(frozen=True, slots=True)
class _Context:
    """Everything one construction needs, pickled once per worker rather than once per build."""

    psalms: tuple[ClausePsalm, ...]
    output_root: Path
    vocabulary: tuple[str, ...]
    external_counts: dict[str, int]
    k: int
    family: ClauseFamily


def write_construction(context: _Context, construction: str) -> str | None:
    """Writes one construction's dataset, or returns None when it is already written."""
    family = context.family
    path = path_to_write(
        context.output_root,
        family.unit,
        construction,
        domain=DATASET_TYPE,
        unit_key="feature",
        level="clause",
    )
    if path is None:
        return None
    args = (list(context.psalms), context.vocabulary, context.external_counts, context.k)
    description = (
        f"{family.description} (RARE-collapsed, k={context.k}, majority colon assignment), "
        f"construction={construction}."
    )
    dense_builder = family.dense.get(construction)
    if dense_builder is not None:
        write_vectors(path, dense_builder(*args), description)
    else:
        sparse_dim = concatenated_1_2_3gram_dim(len(context.vocabulary))
        write_sparse_vectors(path, family.sparse[construction](*args), sparse_dim, description)
    return f"{family.unit}_{construction}"


def generate_family(
    family: ClauseFamily,
    psalms: list[ClausePsalm],
    output_root: Path,
    external_counts: dict[str, int],
    k: int,
    *,
    max_workers: int | None = None,
) -> list[str]:
    """Writes every not-yet-written construction of `family`, returns the names written."""
    context = _Context(
        psalms=tuple(psalms),
        output_root=output_root,
        vocabulary=build_signature_vocabulary(external_counts, k),
        external_counts=external_counts,
        k=k,
        family=family,
    )
    constructions = (*family.dense, *family.sparse)
    return map_constructions(write_construction, context, constructions, max_workers=max_workers)
