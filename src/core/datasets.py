"""Names, finds, and reads the datasets this package writes, for every consumer of the tree."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

TEXT_VARIANTS = ("consonantal", "vocalized", "cantillation")

#: One node's nonzero entries, as the lists Parquet returns or as the arrays a builder produces.
type SparseRows = Sequence[Sequence[float] | np.ndarray]

#: A trailing draw number marks one sample of the order-shuffle null, not a dataset.
_SHUFFLE_DRAW = re.compile(r"shuffle\d+$")

#: Rows decoded per step, since Arrow's decode peak is a multiple of the slice it decodes.
DENSE_BATCH_ROWS = 256


class UnnamedDatasetError(ValueError):
    """A Parquet file outside any Hive partition names no dataset."""


def dataset_identifier(path: Path) -> str:
    """Joins every Hive partition value between the file and its `domain=` root, deepest first."""
    parts = []
    node = path.parent
    while "=" in node.name and not node.name.startswith("domain="):
        parts.append(node.name.split("=", 1)[1])
        node = node.parent
    #: An unpartitioned file names no dataset, and a blank name would reach an output as a row.
    if not parts:
        raise UnnamedDatasetError(f"{path} carries no Hive partition, so it names no dataset")
    return "_".join(reversed(parts))


def split_model_name(model: str, text_variants: tuple[str, ...] = TEXT_VARIANTS) -> tuple[str, str]:
    """Splits into (base_model, text_variant): a trailing suffix, else a variant token anywhere."""
    for variant in text_variants:
        suffix = f"_{variant}"
        if model.endswith(suffix):
            return model[: -len(suffix)].removeprefix("semantic_"), variant
    tokens = model.split("_")
    for variant in text_variants:
        if variant in tokens:
            base = "_".join(t for t in tokens if t != variant)
            return base, variant
    return model.removeprefix("semantic_"), "unknown"


def discover_domains(root: Path) -> tuple[str, ...]:
    """The representation domains present in the tree."""
    return tuple(sorted(p.name.split("=", 1)[1] for p in root.glob("domain=*") if p.is_dir()))


def is_shuffle_draw(path: Path) -> bool:
    """True when a file is one draw of the order-shuffle null rather than a dataset."""
    return any(_SHUFFLE_DRAW.search(part.partition("=")[2] or part) for part in path.parts)


def names_a_dataset(path: Path) -> bool:
    """A parquet outside a Hive partition tree is some other output, so a sweep passes it over."""
    parent = path.parent.name
    return "=" in parent and not parent.startswith("domain=")


def dataset_paths(root: Path) -> list[Path]:
    """Every dataset file under the root, in canonical order, excluding order-shuffle draws."""
    return [
        path
        for path in sorted(root.glob("**/*.parquet"))
        if path.is_file() and not is_shuffle_draw(path) and names_a_dataset(path)
    ]


def is_sparse_embeddings(path: Path) -> bool:
    """Sparse files carry indices/values in place of a dense vector column."""
    return "vector" not in pq.read_schema(path).names


def release_arrow_memory() -> None:
    """Hands Arrow's freed buffers back to the OS, so a batch over many files stays flat."""
    pa.default_memory_pool().release_unused()


def read_dense_rows(path: Path, batch_size: int = DENSE_BATCH_ROWS) -> dict[int, np.ndarray]:
    """Every row of a dense file, decoded in batches into one float32 matrix, zero vectors kept."""
    reader = pq.ParquetFile(path)
    dim = reader.schema_arrow.field("vector").type.list_size
    matrix = np.empty((reader.metadata.num_rows, dim), dtype="<f4")
    node_ids = np.empty(reader.metadata.num_rows, dtype=np.int64)
    start = 0
    for batch in reader.iter_batches(batch_size=batch_size, columns=["node_id", "vector"]):
        stop = start + batch.num_rows
        node_ids[start:stop] = batch.column("node_id").to_numpy(zero_copy_only=False)
        values = batch.column("vector").values.to_numpy(zero_copy_only=False)
        matrix[start:stop] = values.reshape(batch.num_rows, dim)
        start = stop
    del reader
    release_arrow_memory()
    return {int(node_ids[i]): matrix[i] for i in range(len(node_ids))}


def sparse_rows_to_csr(
    node_ids: list[int], indices_col: SparseRows, values_col: SparseRows, dim: int
) -> tuple[list[int], sp.csr_matrix]:
    """One CSR matrix from per-node index and value rows, keeping a row that carries none."""
    row_lengths = [len(indices) for indices in indices_col]
    indptr = np.concatenate([[0], np.cumsum(row_lengths)])
    flat_indices = (
        np.concatenate([np.asarray(idx, dtype=np.int32) for idx in indices_col])
        if any(row_lengths)
        else np.zeros(0, dtype=np.int32)
    )
    flat_values = (
        np.concatenate([np.asarray(val, dtype="<f4") for val in values_col])
        if any(row_lengths)
        else np.zeros(0, dtype="<f4")
    )
    return node_ids, sp.csr_matrix((flat_values, flat_indices, indptr), shape=(len(node_ids), dim))


def read_sparse_rows(path: Path) -> tuple[list[int], sp.csr_matrix]:
    """Every row of a sparse file, keeping a half-verse carrying no nonzeros as the value it is."""
    table = pq.read_table(path, columns=["node_id", "indices", "values"])
    rows = sparse_rows_to_csr(
        table["node_id"].to_pylist(),
        table["indices"].to_pylist(),
        table["values"].to_pylist(),
        int(table.schema.metadata[b"dim"]),
    )
    del table
    release_arrow_memory()
    return rows
