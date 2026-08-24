"""Writes one embedding matrix as a Parquet file, keyed by BHSA colon node id."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from semantic.corpus import Psalm

DATASET_VERSION = "1.0"


def dataset_path(output_root: Path, model: str, variation: str) -> Path:
    """Returns the Hive-partitioned `.parquet` file path for a model and variation."""
    return (
        output_root / "domain=semantic" / f"model={model}" / f"text={variation}" / "part-0.parquet"
    )


def node_vectors(embeddings: dict[int, np.ndarray], psalms: list[Psalm]) -> dict[int, np.ndarray]:
    """Maps each psalm's embedding vectors to its BHSA colon node ids."""
    values: dict[int, np.ndarray] = {}
    for psalm in psalms:
        vectors = embeddings.get(psalm.number)
        if vectors is None:
            continue
        for node, vector in zip(psalm.colon_nodes, vectors, strict=True):
            values[node] = vector
    return values


def write_dataset(
    output_root: Path, model: str, variation: str, vectors: dict[int, np.ndarray], description: str
) -> None:
    """Writes one Parquet file: columns node_id (int32) and vector (float32 list)."""
    path = dataset_path(output_root, model, variation)
    path.parent.mkdir(parents=True, exist_ok=True)
    node_ids = sorted(vectors)
    dim = len(vectors[node_ids[0]])
    matrix = np.stack([vectors[n].astype("<f4") for n in node_ids])
    table = pa.table(
        {
            "node_id": pa.array(node_ids, type=pa.int32()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(matrix.flatten(), type=pa.float32()), dim
            ),
        }
    )
    table = table.replace_schema_metadata({"description": description, "version": DATASET_VERSION})
    pq.write_table(table, path, compression="zstd")
