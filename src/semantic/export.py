"""Where one model's embeddings of one source at one tier are written."""

from __future__ import annotations

from pathlib import Path

from core.export import DATASET_VERSION, skip_if_written
from core.partition import BHSA_HALF_VERSE, Partition, Scope
from semantic import DOMAIN

__all__ = ["DATASET_VERSION", "partition", "path_to_write"]


def partition(model: str, variation: str, scope: Scope = BHSA_HALF_VERSE) -> Partition:
    """The partition one model's embeddings at one text tier are written under."""
    return Partition(scope, DOMAIN, model=model, text=variation)


def path_to_write(
    output_root: Path, model: str, variation: str, scope: Scope = BHSA_HALF_VERSE
) -> Path | None:
    """The path this model and variation belong at, or None when it is already written."""
    target = partition(model, variation, scope)
    return skip_if_written(target.file(output_root), target.directory)
