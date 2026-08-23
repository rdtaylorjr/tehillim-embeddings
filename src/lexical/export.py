"""Writes one lexical vector matrix as a Parquet file, keyed by BHSA half-verse node id."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATASET_VERSION = "1.0"


def dataset_path(
    output_root: Path,
    vocab: str,
    weight: str,
    text: str | None = None,
    dataset_type: str = "lexical",
) -> Path:
    """Hive-partitioned `.parquet` path for a unit/construction, with an optional text tier."""
    text_segment = (f"text={text}",) if text is not None else ()
    return (
        output_root.joinpath(
            f"type={dataset_type}",
            f"unit={vocab}",
            *text_segment,
            f"construction={weight}",
        )
        / "part-0.parquet"
    )


def write_dataset(
    output_root: Path,
    vocab: str,
    weight: str,
    vectors: dict[int, np.ndarray],
    description: str,
    text: str | None = None,
    dataset_type: str = "lexical",
) -> None:
    """Writes one Parquet file: columns node_id (int32) and vector (float32 list)."""
    path = dataset_path(output_root, vocab, weight, text=text, dataset_type=dataset_type)
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


def write_sparse_dataset(
    output_root: Path,
    vocab: str,
    weight: str,
    sparse_vectors: dict[int, tuple[np.ndarray, np.ndarray]],
    dim: int,
    description: str,
    text: str | None = None,
    dataset_type: str = "lexical",
) -> None:
    """Writes one sparse Parquet file: node_id, indices (list<int32>), values (list<float32>)."""
    path = dataset_path(output_root, vocab, weight, text=text, dataset_type=dataset_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    node_ids = sorted(sparse_vectors)
    indices_col = [sparse_vectors[n][0].astype("<i4").tolist() for n in node_ids]
    values_col = [sparse_vectors[n][1].astype("<f4").tolist() for n in node_ids]
    table = pa.table(
        {
            "node_id": pa.array(node_ids, type=pa.int32()),
            "indices": pa.array(indices_col, type=pa.list_(pa.int32())),
            "values": pa.array(values_col, type=pa.list_(pa.float32())),
        }
    )
    table = table.replace_schema_metadata(
        {
            "description": description,
            "version": DATASET_VERSION,
            "dim": str(dim),
            "sparse": "true",
        }
    )
    pq.write_table(table, path, compression="zstd")
