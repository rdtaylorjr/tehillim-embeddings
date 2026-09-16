"""One writer for every dataset family: build each construction, skip what exists, parallelize."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.export import path_to_write, write_sparse_vectors, write_vectors
from core.parallel import map_constructions
from core.partition import BHSA_HALF_VERSE, Partition, Scope

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
class _Job[PsalmT]:
    """Everything one construction needs, pickled once per worker rather than once per build."""

    psalms: tuple[PsalmT, ...]
    output_root: Path
    scope: Scope
    domain: str
    feature: str
    level: str | None

    def partition(self, construction: str) -> Partition:
        """The partition one construction of the family is written at."""
        return Partition(
            self.scope,
            self.domain,
            level=self.level,
            feature=self.feature,
            construction=construction,
        )


def write_construction[PsalmT](
    job: _Job[PsalmT], item: tuple[str, Construction[PsalmT]]
) -> str | None:
    """Writes one construction's dataset, or returns None when it is already written."""
    name, construction = item
    partition = job.partition(name)
    path = path_to_write(job.output_root, partition)
    if path is None:
        return None
    vectors = construction.build(list(job.psalms))
    if construction.sparse_dim is None:
        write_vectors(path, vectors, construction.description)
    else:
        write_sparse_vectors(path, vectors, construction.sparse_dim, construction.description)
    return partition.identifier


def generate_family[PsalmT](
    psalms: Sequence[PsalmT],
    output_root: Path,
    constructions: Sequence[tuple[str, Construction[PsalmT]]],
    *,
    domain: str,
    feature: str,
    level: str | None = None,
    scope: Scope = BHSA_HALF_VERSE,
    max_workers: int | None = None,
) -> list[str]:
    """Writes each not-yet-written construction of one family, returns the names written."""
    job = _Job(tuple(psalms), output_root, scope, domain, feature, level)
    return map_constructions(write_construction, job, constructions, max_workers=max_workers)
