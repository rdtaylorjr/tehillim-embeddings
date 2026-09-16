"""Writes one embedding matrix as a Parquet file under its partition, keyed by node id."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from core.export import DATASET_VERSION, skip_if_written, write_vectors
from core.partition import BHSA_HALF_VERSE, Partition, Scope
from semantic import DOMAIN

if TYPE_CHECKING:
    from semantic.corpus import SemanticPsalm

__all__ = ["DATASET_VERSION", "node_vectors", "partition", "path_to_write", "write_dataset"]


def partition(model: str, variation: str, scope: Scope = BHSA_HALF_VERSE) -> Partition:
    """The partition one model's embeddings at one text tier are written under."""
    return Partition(scope, DOMAIN, model=model, text=variation)


def path_to_write(output_root: Path, model: str, variation: str) -> Path | None:
    """The path this model and variation belong at, or None when it is already written."""
    return skip_if_written(
        partition(model, variation).file(output_root), f"semantic model={model} text={variation}"
    )


def node_vectors(
    embeddings: dict[int, np.ndarray], psalms: list[SemanticPsalm]
) -> dict[int, np.ndarray]:
    """Maps each psalm's embedding vectors to its BHSA half-verse node ids."""
    return {
        node: vector
        for psalm in psalms
        if (vectors := embeddings.get(psalm.number)) is not None
        for node, vector in zip(psalm.half_verse_nodes, vectors, strict=True)
    }


def write_dataset(
    output_root: Path, model: str, variation: str, vectors: dict[int, np.ndarray], description: str
) -> None:
    """Writes one Parquet file: columns node_id (int32) and vector (float32 list)."""
    write_vectors(partition(model, variation).file(output_root), vectors, description)
