"""One feature family whose vocabulary is built from a frozen external-support table."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.dataset_family import Construction, generate_family
from core.ngram import concatenated_1_2_3gram_dim
from core.support import build_signature_vocabulary

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

#: Every builder in a supported family reads the psalms, the vocabulary, its counts and its k.
type SupportedBuilder = Callable[..., dict[int, Any]]


@dataclass(frozen=True, slots=True)
class SupportedDataset[PsalmT]:
    """Everything that separates one support-table family's datasets from another's."""

    unit: str
    domain: str
    dense: Mapping[str, SupportedBuilder]
    sparse: Mapping[str, SupportedBuilder]
    #: Formatted with the frozen k, then completed with the construction being written.
    description: str
    level: str | None = None


def generate_supported_dataset[PsalmT](
    dataset: SupportedDataset[PsalmT],
    psalms: Sequence[PsalmT],
    output_root: Path,
    external_counts: dict[str, int],
    k: int,
    *,
    max_workers: int | None = None,
) -> list[str]:
    """Writes every not-yet-written construction of one supported family, returns the names."""
    vocabulary = build_signature_vocabulary(external_counts, k)
    prefix = dataset.description.format(k=k)
    dimension = concatenated_1_2_3gram_dim(len(vocabulary))
    family = [
        (
            name,
            Construction(
                build=partial(builder, vocabulary=vocabulary, external_counts=external_counts, k=k),
                description=f"{prefix}, construction={name}.",
                sparse_dim=dimension if sparse else None,
            ),
        )
        for table, sparse in ((dataset.dense, False), (dataset.sparse, True))
        for name, builder in table.items()
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
