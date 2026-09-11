"""One n-gram feature's dataset family: where it is written, what it reads, and at which orders."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from core.dataset_family import Builder, Construction, generate_family
from core.ngram import (
    ColumnsOf,
    concatenated_1_2_3gram_dim,
    ngram_psalm_vectors,
    ngram_vectors,
    sparse_ngram_psalm_vectors,
    sparse_ngram_vectors,
)
from core.shuffle import NumberedPsalm

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class NgramDataset[PsalmT: NumberedPsalm]:
    """Everything that separates one feature's n-gram datasets from another's."""

    unit: str
    domain: str
    columns_of: ColumnsOf[PsalmT]
    vocabulary: tuple[str, ...]
    #: Construction name to the orders it concatenates, in the order the datasets are written.
    constructions: Mapping[str, tuple[int, ...]]
    description: str
    #: The constructions whose trigram block is too sparse to store densely.
    sparse: frozenset[str] = field(default_factory=frozenset)
    level: str | None = None


def construction_of[PsalmT: NumberedPsalm](
    dataset: NgramDataset[PsalmT], name: str, orders: tuple[int, ...]
) -> Construction[PsalmT]:
    """One construction of a feature, bound to its builder, its width and its description."""
    pooled = name.endswith("_psalm")
    width = len(dataset.vocabulary)
    if name in dataset.sparse:
        sparse_build = sparse_ngram_psalm_vectors if pooled else sparse_ngram_vectors
        dimension = concatenated_1_2_3gram_dim(width)
        bound: Builder[PsalmT] = partial(
            sparse_build, columns_of=dataset.columns_of, vocabulary=dataset.vocabulary
        )
    else:
        dense_build = ngram_psalm_vectors if pooled else ngram_vectors
        dimension = sum(width**order for order in orders)
        bound = partial(
            dense_build,
            columns_of=dataset.columns_of,
            vocabulary=dataset.vocabulary,
            orders=orders,
        )
    return Construction(
        build=bound,
        description=f"{dataset.description}, construction={name}, dimension {dimension}.",
        sparse_dim=dimension if name in dataset.sparse else None,
    )


def order_sensitive_constructions[PsalmT: NumberedPsalm](
    dataset: NgramDataset[PsalmT],
) -> list[tuple[str, Construction[PsalmT]]]:
    """The constructions a half-verse shuffle can move: those concatenating a bigram or higher."""
    return [
        (name, construction_of(dataset, name, orders))
        for name, orders in dataset.constructions.items()
        if max(orders) >= 2
    ]


def generate_ngram_dataset[PsalmT: NumberedPsalm](
    dataset: NgramDataset[PsalmT],
    psalms: Sequence[PsalmT],
    output_root: Path,
    *,
    max_workers: int | None = None,
) -> list[str]:
    """Writes every not-yet-written construction of one feature, returns the names written."""
    family = [
        (name, construction_of(dataset, name, orders))
        for name, orders in dataset.constructions.items()
    ]
    return generate_family(
        psalms,
        output_root,
        family,
        unit=dataset.unit,
        domain=dataset.domain,
        level=dataset.level,
        max_workers=max_workers,
    )
