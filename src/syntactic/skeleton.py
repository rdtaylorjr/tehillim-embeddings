"""Shared driver for the syntactic skeleton generators, each a per-node and psalm pair."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.export import path_to_write, write_vectors
from core.parallel import map_constructions
from syntactic import DATASET_TYPE

type Builder[RecordT] = Callable[[list[RecordT]], dict[int, np.ndarray]]


@dataclass(frozen=True, slots=True)
class _SkeletonContext[RecordT]:
    """Everything one construction needs, pickled once per worker rather than once per build."""

    psalms: tuple[RecordT, ...]
    output_root: Path
    unit: str
    description: str
    level: str


def write_construction[RecordT](
    context: _SkeletonContext[RecordT], item: tuple[str, Builder[RecordT]]
) -> str | None:
    """Writes one construction's dataset, or returns None when it is already written."""
    construction, builder = item
    path = path_to_write(
        context.output_root,
        context.unit,
        construction,
        domain=DATASET_TYPE,
        unit_key="feature",
        level=context.level,
    )
    if path is None:
        return None
    description = f"{context.description}, construction={construction}."
    write_vectors(path, builder(list(context.psalms)), description)
    return f"{context.unit}_{construction}"


def generate_skeleton[RecordT](
    psalms: list[RecordT],
    output_root: Path,
    unit: str,
    constructions: Sequence[tuple[str, Builder[RecordT]]],
    description: str,
    *,
    level: str,
    max_workers: int | None = None,
) -> list[str]:
    """Writes each not-yet-written `<unit>` construction, returns the qualified names written."""
    context = _SkeletonContext(tuple(psalms), output_root, unit, description, level)
    return map_constructions(write_construction, context, constructions, max_workers=max_workers)
