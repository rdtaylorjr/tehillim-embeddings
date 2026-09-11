"""Binds the syntactic domain to the shared dataset writer, for the per-node and psalm pairs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from core.dataset_family import Builder, Construction, generate_family
from syntactic import DATASET_TYPE

if TYPE_CHECKING:
    from collections.abc import Sequence


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
    family = [
        (name, Construction(build=builder, description=f"{description}, construction={name}."))
        for name, builder in constructions
    ]
    return generate_family(
        psalms,
        output_root,
        family,
        unit=unit,
        domain=DATASET_TYPE,
        level=level,
        max_workers=max_workers,
    )
