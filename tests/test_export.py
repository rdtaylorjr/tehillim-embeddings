from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq

from semantic.export import dataset_path, node_vectors, write_dataset


def _psalm(*, number: int, cola, colon_nodes):
    from semantic.corpus import Psalm

    return Psalm(number=number, cola=cola, colon_nodes=colon_nodes)


class TestDatasetPath:
    def test_points_at_the_hive_partitioned_parquet_file(self, tmp_path):
        path = dataset_path(tmp_path, "bge_m3", "vocalized")
        expected = (
            tmp_path / "domain=semantic" / "model=bge_m3" / "text=vocalized" / "part-0.parquet"
        )
        assert path == expected


class TestNodeVectors:
    def test_maps_each_colon_node_to_its_vector(self):
        psalms = [_psalm(number=1, cola=("A", "B"), colon_nodes=(100, 101))]
        embeddings = {1: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}

        values = node_vectors(embeddings, psalms)

        assert set(values) == {100, 101}
        assert np.array_equal(values[100], [1.0, 2.0])
        assert np.array_equal(values[101], [3.0, 4.0])

    def test_skips_a_psalm_missing_from_the_embeddings_dict(self):
        psalms = [_psalm(number=1, cola=("A",), colon_nodes=(100,))]

        values = node_vectors({}, psalms)

        assert values == {}


class TestWriteDataset:
    def test_round_trips_node_ids_and_vectors_exactly(self, tmp_path):
        vectors = {
            100: np.array([1.0, -2.5, 0.0], dtype=np.float32),
            101: np.array([3.25, 4.0, -1.0], dtype=np.float32),
        }

        write_dataset(tmp_path, "bge_m3", "vocalized", vectors, "a description")

        table = pq.read_table(dataset_path(tmp_path, "bge_m3", "vocalized"))
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert set(by_node) == {100, 101}
        assert np.allclose(by_node[100], vectors[100])
        assert np.allclose(by_node[101], vectors[101])

    def test_stores_float32_regardless_of_input_dtype(self, tmp_path):
        vectors = {100: np.array([1.0, 2.0], dtype=np.float64)}

        write_dataset(tmp_path, "bge_m3", "vocalized", vectors, "a description")

        table = pq.read_table(dataset_path(tmp_path, "bge_m3", "vocalized"))
        assert (
            table["vector"].type.value_type == "float32"
            or str(table["vector"].type.value_type) == "float"
        )

    def test_stores_the_description_in_file_metadata(self, tmp_path):
        vectors = {100: np.array([1.0], dtype=np.float32)}

        write_dataset(tmp_path, "bge_m3", "vocalized", vectors, "a specific description")

        table = pq.read_table(dataset_path(tmp_path, "bge_m3", "vocalized"))
        assert table.schema.metadata[b"description"] == b"a specific description"

    def test_creates_parent_directories(self, tmp_path):
        vectors = {100: np.array([1.0], dtype=np.float32)}

        write_dataset(tmp_path / "nested", "bge_m3", "vocalized", vectors, "d")

        assert dataset_path(tmp_path / "nested", "bge_m3", "vocalized").exists()
