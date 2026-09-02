"""Writes one embedding matrix as a Parquet file, keyed by BHSA half-verse node id."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from core.export import DATASET_VERSION, write_vectors

if TYPE_CHECKING:
    from semantic.corpus import Psalm

__all__ = ["DATASET_VERSION", "dataset_path", "node_vectors", "write_dataset"]


def dataset_path(output_root: Path, model: str, variation: str) -> Path:
    """Returns the Hive-partitioned `.parquet` file path for a model and variation."""
    return (
        output_root / "domain=semantic" / f"model={model}" / f"text={variation}" / "part-0.parquet"
    )


def node_vectors(embeddings: dict[int, np.ndarray], psalms: list[Psalm]) -> dict[int, np.ndarray]:
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
    write_vectors(dataset_path(output_root, model, variation), vectors, description)
