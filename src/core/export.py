"""Parquet writers shared by every domain: dense and sparse, keyed by the corpus's node id."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from core.partition import Partition

DATASET_VERSION = "1.0"


def _sorted_nodes(vectors: Mapping[int, object], path: Path) -> list[int]:
    """Node ids in ascending order, rejecting an empty mapping before any file is created."""
    if not vectors:
        raise ValueError(f"refusing to write {path}: no vectors to write")
    return sorted(vectors)


def write_vectors(path: Path, vectors: dict[int, np.ndarray], description: str) -> None:
    """Writes one dense Parquet file: node_id (int32) and vector (fixed-size float32 list)."""
    node_ids = _sorted_nodes(vectors, path)
    lengths = {len(vectors[node]) for node in node_ids}
    if len(lengths) > 1:
        raise ValueError(
            f"refusing to write {path}: vectors must all be the same length, got {sorted(lengths)}"
        )
    dim = lengths.pop()
    #: Filled in place rather than stacked from a list, which would hold two copies at once.
    matrix = np.empty((len(node_ids), dim), dtype="<f4")
    for row, node in enumerate(node_ids):
        matrix[row] = vectors[node]
    table = pa.table(
        {
            "node_id": pa.array(node_ids, type=pa.int32()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(matrix.ravel(), type=pa.float32()), dim
            ),
        }
    )
    _write(table, path, {"description": description, "version": DATASET_VERSION})


def write_sparse_vectors(
    path: Path,
    sparse_vectors: dict[int, tuple[np.ndarray, np.ndarray]],
    dim: int,
    description: str,
) -> None:
    """Writes one sparse Parquet file: node_id, indices (list<int32>), values (list<float32>)."""
    node_ids = _sorted_nodes(sparse_vectors, path)
    #: One batched bounds check over all indices, naming the offending node only on failure.
    all_indices = np.concatenate([sparse_vectors[node][0] for node in node_ids])
    if all_indices.size and (all_indices.max() >= dim or all_indices.min() < 0):
        offender = next(
            node
            for node in node_ids
            if (indices := sparse_vectors[node][0]).size
            and (indices.max() >= dim or indices.min() < 0)
        )
        raise ValueError(
            f"refusing to write {path}: node {offender} has an index outside [0, {dim})"
        )
    #: Two flat buffers over shared offsets, so no row is ever built as a Python list.
    all_values = np.concatenate([sparse_vectors[node][1] for node in node_ids])
    lengths = np.fromiter(
        (sparse_vectors[node][0].size for node in node_ids), dtype=np.int32, count=len(node_ids)
    )
    offsets = np.zeros(len(node_ids) + 1, dtype=np.int32)
    np.cumsum(lengths, out=offsets[1:])
    offset_array = pa.array(offsets, type=pa.int32())
    table = pa.table(
        {
            "node_id": pa.array(node_ids, type=pa.int32()),
            "indices": pa.ListArray.from_arrays(
                offset_array, pa.array(all_indices.astype("<i4"), type=pa.int32())
            ),
            "values": pa.ListArray.from_arrays(
                offset_array, pa.array(all_values.astype("<f4"), type=pa.float32())
            ),
        }
    )
    _write(
        table,
        path,
        {
            "description": description,
            "version": DATASET_VERSION,
            "dim": str(dim),
            "sparse": "true",
        },
    )


def _write(table: pa.Table, path: Path, metadata: dict[str, str]) -> None:
    """Stamps the partition's keys and the schema metadata, then writes the table."""
    #: The file carries the keys its path spells, so the two cannot disagree.
    stamped = {**metadata, **Partition.parse(path).metadata()}
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(stamped), path, compression="zstd")


def skip_if_written(path: Path, label: str) -> Path | None:
    """The path to write, or None when the dataset already exists and the build is skipped."""
    if path.exists():
        return None
    print(f"computing {label}...", file=sys.stderr)
    return path


def path_to_write(output_root: Path, partition: Partition) -> Path | None:
    """The file a partition belongs at, or None when it is already written."""
    return skip_if_written(partition.file(output_root), partition.directory)


def write_dataset(
    output_root: Path, partition: Partition, vectors: dict[int, np.ndarray], description: str
) -> None:
    """Writes one partition's dense Parquet file."""
    write_vectors(partition.file(output_root), vectors, description)


def write_sparse_dataset(
    output_root: Path,
    partition: Partition,
    sparse_vectors: dict[int, tuple[np.ndarray, np.ndarray]],
    dim: int,
    description: str,
) -> None:
    """Writes one partition's sparse Parquet file."""
    write_sparse_vectors(partition.file(output_root), sparse_vectors, dim, description)
