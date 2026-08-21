from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq

from lexical.export import dataset_path, write_dataset


class TestDatasetPath:
    def test_points_at_the_hive_partitioned_parquet_file(self, tmp_path):
        path = dataset_path(tmp_path, "homograph", "binary")
        expected = (
            tmp_path
            / "data"
            / "type=lexical"
            / "vocab=homograph"
            / "weight=binary"
            / "part-0.parquet"
        )
        assert path == expected

    def test_inserts_a_text_partition_when_given(self, tmp_path):
        path = dataset_path(tmp_path, "word", "binary", text="consonantal")
        expected = (
            tmp_path
            / "data"
            / "type=lexical"
            / "vocab=word"
            / "text=consonantal"
            / "weight=binary"
            / "part-0.parquet"
        )
        assert path == expected


class TestWriteDataset:
    def test_round_trips_node_ids_and_vectors_exactly(self, tmp_path):
        vectors = {
            100: np.array([1.0, 0.0, 1.0], dtype=np.float32),
            101: np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }

        write_dataset(tmp_path, "homograph", "binary", vectors, "a description")

        table = pq.read_table(dataset_path(tmp_path, "homograph", "binary"))
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert set(by_node) == {100, 101}
        assert np.allclose(by_node[100], vectors[100])
        assert np.allclose(by_node[101], vectors[101])

    def test_stores_float32_regardless_of_input_dtype(self, tmp_path):
        vectors = {100: np.array([1.0, 0.0], dtype=np.float64)}

        write_dataset(tmp_path, "homograph", "binary", vectors, "a description")

        table = pq.read_table(dataset_path(tmp_path, "homograph", "binary"))
        assert (
            table["vector"].type.value_type == "float32"
            or str(table["vector"].type.value_type) == "float"
        )

    def test_stores_the_description_in_file_metadata(self, tmp_path):
        vectors = {100: np.array([1.0], dtype=np.float32)}

        write_dataset(tmp_path, "homograph", "binary", vectors, "a specific description")

        table = pq.read_table(dataset_path(tmp_path, "homograph", "binary"))
        assert table.schema.metadata[b"description"] == b"a specific description"

    def test_creates_parent_directories(self, tmp_path):
        vectors = {100: np.array([1.0], dtype=np.float32)}

        write_dataset(tmp_path / "nested", "homograph", "binary", vectors, "d")

        assert dataset_path(tmp_path / "nested", "homograph", "binary").exists()

    def test_writes_to_the_text_partitioned_path_when_given(self, tmp_path):
        vectors = {100: np.array([1.0], dtype=np.float32)}

        write_dataset(tmp_path, "word", "binary", vectors, "d", text="vocalized")

        assert dataset_path(tmp_path, "word", "binary", text="vocalized").exists()
