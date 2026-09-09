"""Shared driver for the phrase skeleton generators, each a per-node and psalm pair."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.export import path_to_write, write_vectors
from core.parallel import map_constructions
from syntactic import DATASET_TYPE
from syntactic.corpus import PhrasePsalm

Builder = Callable[[list[PhrasePsalm]], dict[int, np.ndarray]]


@dataclass(frozen=True, slots=True)
class _SkeletonContext:
    """Everything one construction needs, pickled once per worker rather than once per build."""

    psalms: tuple[PhrasePsalm, ...]
    output_root: Path
    unit: str
    description: str


def write_construction(context: _SkeletonContext, item: tuple[str, Builder]) -> str | None:
    """Writes one construction's dataset, or returns None when it is already written."""
    construction, builder = item
    path = path_to_write(
        context.output_root,
        context.unit,
        construction,
        domain=DATASET_TYPE,
        unit_key="feature",
        level="phrase",
    )
    if path is None:
        return None
    description = f"{context.description}, construction={construction}."
    write_vectors(path, builder(list(context.psalms)), description)
    return f"{context.unit}_{construction}"


def generate_skeleton(
    psalms: list[PhrasePsalm],
    output_root: Path,
    unit: str,
    constructions: Sequence[tuple[str, Builder]],
    description: str,
    *,
    max_workers: int | None = None,
) -> list[str]:
    """Writes each not-yet-written `<unit>` construction, returns the qualified names written."""
    context = _SkeletonContext(tuple(psalms), output_root, unit, description)
    return map_constructions(write_construction, context, constructions, max_workers=max_workers)
