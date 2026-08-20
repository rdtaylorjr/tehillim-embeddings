from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq

from lexical.export import dataset_path, write_dataset


class TestDatasetPath:
    def test_points_at_the_hive_partitioned_parquet_file(self, tmp_path):
        path = dataset_path(tmp_path, "lex0", "binary")
        expected = (
            tmp_path / "data" / "type=lexical" / "vocab=lex0" / "weight=binary" / "part-0.parquet"
        )
        assert path == expected


class TestWriteDataset:
    def test_round_trips_node_ids_and_vectors_exactly(self, tmp_path):
        vectors = {
            100: np.array([1.0, 0.0, 1.0], dtype=np.float32),
            101: np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }

        write_dataset(tmp_path, "lex0", "binary", vectors, "a description")

        table = pq.read_table(dataset_path(tmp_path, "lex0", "binary"))
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert set(by_node) == {100, 101}
        assert np.allclose(by_node[100], vectors[100])
        assert np.allclose(by_node[101], vectors[101])

    def test_stores_float32_regardless_of_input_dtype(self, tmp_path):
        vectors = {100: np.array([1.0, 0.0], dtype=np.float64)}

        write_dataset(tmp_path, "lex0", "binary", vectors, "a description")

        table = pq.read_table(dataset_path(tmp_path, "lex0", "binary"))
        assert (
            table["vector"].type.value_type == "float32"
            or str(table["vector"].type.value_type) == "float"
        )

    def test_stores_the_description_in_file_metadata(self, tmp_path):
        vectors = {100: np.array([1.0], dtype=np.float32)}

        write_dataset(tmp_path, "lex0", "binary", vectors, "a specific description")

        table = pq.read_table(dataset_path(tmp_path, "lex0", "binary"))
        assert table.schema.metadata[b"description"] == b"a specific description"

    def test_creates_parent_directories(self, tmp_path):
        vectors = {100: np.array([1.0], dtype=np.float32)}

        write_dataset(tmp_path / "nested", "lex0", "binary", vectors, "d")

        assert dataset_path(tmp_path / "nested", "lex0", "binary").exists()
