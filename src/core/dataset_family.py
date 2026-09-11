"""One writer for every dataset family: build each construction, skip what exists, parallelize."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.export import path_to_write, write_sparse_vectors, write_vectors
from core.parallel import map_constructions

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

#: Builds one construction's vectors from the psalms, with everything else already bound.
type Builder[PsalmT] = Callable[[list[PsalmT]], dict[int, Any]]


@dataclass(frozen=True, slots=True)
class Construction[PsalmT]:
    """One dataset to write: how to build it, what to record, and how it is stored."""

    build: Builder[PsalmT]
    description: str
    #: The dense width when the vectors are stored as sparse pairs, None when stored densely.
    sparse_dim: int | None = None


@dataclass(frozen=True, slots=True)
class _Partition[PsalmT]:
    """Everything one construction needs, pickled once per worker rather than once per build."""

    psalms: tuple[PsalmT, ...]
    output_root: Path
    unit: str
    domain: str
    level: str | None


def write_construction[PsalmT](
    partition: _Partition[PsalmT], item: tuple[str, Construction[PsalmT]]
) -> str | None:
    """Writes one construction's dataset, or returns None when it is already written."""
    name, construction = item
    path = path_to_write(
        partition.output_root,
        partition.unit,
        name,
        domain=partition.domain,
        unit_key="feature",
        level=partition.level,
    )
    if path is None:
        return None
    vectors = construction.build(list(partition.psalms))
    if construction.sparse_dim is None:
        write_vectors(path, vectors, construction.description)
    else:
        write_sparse_vectors(path, vectors, construction.sparse_dim, construction.description)
    return f"{partition.unit}_{name}"


def generate_family[PsalmT](
    psalms: Sequence[PsalmT],
    output_root: Path,
    constructions: Sequence[tuple[str, Construction[PsalmT]]],
    *,
    unit: str,
    domain: str,
    level: str | None = None,
    max_workers: int | None = None,
) -> list[str]:
    """Writes each not-yet-written construction of one family, returns the names written."""
    partition = _Partition(tuple(psalms), output_root, unit, domain, level)
    return map_constructions(write_construction, partition, constructions, max_workers=max_workers)
