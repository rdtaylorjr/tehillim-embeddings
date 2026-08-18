"""Writes one lexical vector matrix as a Parquet file, keyed by BHSA half-verse node id."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATASET_VERSION = "1.0"


def dataset_path(output_root: Path, vocab: str, weight: str) -> Path:
    """Returns the Hive-partitioned `.parquet` file path for a vocabulary and weighting scheme."""
    return (
        output_root
        / "data"
        / "type=lexical"
        / f"vocab={vocab}"
        / f"weight={weight}"
        / "part-0.parquet"
    )


def write_dataset(
    output_root: Path, vocab: str, weight: str, vectors: dict[int, np.ndarray], description: str
) -> None:
    """Writes one Parquet file: columns node_id (int32) and vector (float32 list)."""
    path = dataset_path(output_root, vocab, weight)
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
