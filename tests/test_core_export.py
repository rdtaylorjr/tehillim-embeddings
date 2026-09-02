from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import pytest

from core.export import DATASET_VERSION, write_sparse_vectors, write_vectors


class TestWriteVectors:
    def test_writes_node_ids_sorted_ascending(self, tmp_path):
        path = tmp_path / "part-0.parquet"

        write_vectors(
            path,
            {30: np.array([1.0, 2.0]), 10: np.array([3.0, 4.0]), 20: np.array([5.0, 6.0])},
            "desc",
        )

        assert pq.read_table(path)["node_id"].to_pylist() == [10, 20, 30]

    def test_pairs_each_node_with_its_own_vector_after_sorting(self, tmp_path):
        path = tmp_path / "part-0.parquet"

        write_vectors(path, {30: np.array([1.0, 2.0]), 10: np.array([3.0, 4.0])}, "desc")

        table = pq.read_table(path)
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert by_node[10] == [3.0, 4.0]
        assert by_node[30] == [1.0, 2.0]

    def test_stores_vectors_as_float32(self, tmp_path):
        path = tmp_path / "part-0.parquet"

        write_vectors(path, {1: np.array([1.5, 2.5], dtype=np.float64)}, "desc")

        assert pq.read_table(path).schema.field("vector").type.value_type == "float"

    def test_records_the_description_and_version_in_schema_metadata(self, tmp_path):
        path = tmp_path / "part-0.parquet"

        write_vectors(path, {1: np.array([1.0])}, "a description")

        metadata = pq.read_table(path).schema.metadata
        assert metadata[b"description"] == b"a description"
        assert metadata[b"version"] == DATASET_VERSION.encode()

    def test_creates_missing_parent_directories(self, tmp_path):
        path = tmp_path / "domain=x" / "unit=y" / "part-0.parquet"

        write_vectors(path, {1: np.array([1.0])}, "desc")

        assert path.exists()

    def test_rejects_an_empty_vector_mapping_with_a_named_error(self, tmp_path):
        with pytest.raises(ValueError, match="no vectors"):
            write_vectors(tmp_path / "part-0.parquet", {}, "desc")

    def test_rejects_ragged_vectors_with_a_named_error(self, tmp_path):
        with pytest.raises(ValueError, match="same length"):
            write_vectors(
                tmp_path / "part-0.parquet",
                {1: np.array([1.0, 2.0]), 2: np.array([1.0])},
                "desc",
            )


class TestWriteSparseVectors:
    def test_writes_indices_and_values_per_node(self, tmp_path):
        path = tmp_path / "part-0.parquet"

        write_sparse_vectors(
            path,
            {
                7: (np.array([0, 3], dtype=np.int32), np.array([1.0, 2.0], dtype=np.float32)),
                4: (np.array([1], dtype=np.int32), np.array([9.0], dtype=np.float32)),
            },
            dim=16,
            description="desc",
        )

        table = pq.read_table(path)
        assert table["node_id"].to_pylist() == [4, 7]
        assert table["indices"].to_pylist() == [[1], [0, 3]]
        assert table["values"].to_pylist() == [[9.0], [1.0, 2.0]]

    def test_records_the_dimension_and_sparse_flag_in_metadata(self, tmp_path):
        path = tmp_path / "part-0.parquet"

        write_sparse_vectors(
            path,
            {1: (np.array([0], dtype=np.int32), np.array([1.0], dtype=np.float32))},
            dim=64,
            description="desc",
        )

        metadata = pq.read_table(path).schema.metadata
        assert metadata[b"dim"] == b"64"
        assert metadata[b"sparse"] == b"true"
        assert metadata[b"version"] == DATASET_VERSION.encode()

    def test_rejects_an_empty_vector_mapping_with_a_named_error(self, tmp_path):
        with pytest.raises(ValueError, match="no vectors"):
            write_sparse_vectors(tmp_path / "part-0.parquet", {}, dim=8, description="desc")

    def test_rejects_an_index_outside_the_declared_dimension(self, tmp_path):
        with pytest.raises(ValueError, match="outside"):
            write_sparse_vectors(
                tmp_path / "part-0.parquet",
                {1: (np.array([9], dtype=np.int32), np.array([1.0], dtype=np.float32))},
                dim=4,
                description="desc",
            )


class TestWriteVectorsPreservesValues:
    def test_float64_input_rounds_exactly_as_astype_float32_does(self, tmp_path):
        path = tmp_path / "part-0.parquet"
        raw = np.array([0.1, 1 / 3, 1e-8, -2.7182818284590452], dtype=np.float64)

        write_vectors(path, {5: raw}, "desc")

        stored = np.array(pq.read_table(path)["vector"].to_pylist()[0], dtype="<f4")
        assert np.array_equal(stored, raw.astype("<f4"))

    def test_round_trips_a_wide_matrix_exactly(self, tmp_path):
        path = tmp_path / "part-0.parquet"
        rng = np.random.default_rng(0)
        vectors = {node: rng.standard_normal(257).astype("<f4") for node in range(400, 460)}

        write_vectors(path, vectors, "desc")

        table = pq.read_table(path)
        nodes = table["node_id"].to_pylist()
        stored = np.array(table["vector"].to_pylist(), dtype="<f4")
        assert nodes == sorted(vectors)
        assert all(np.array_equal(stored[i], vectors[n]) for i, n in enumerate(nodes))

    def test_a_float64_vector_is_not_widened_in_the_file(self, tmp_path):
        path = tmp_path / "part-0.parquet"

        write_vectors(path, {1: np.array([1.5, 2.5], dtype=np.float64)}, "desc")

        assert pq.read_schema(path).field("vector").type.value_type == "float"


class TestWriteSparseVectorsPreservesValues:
    def test_a_row_with_no_nonzero_entries_is_written_as_an_empty_list(self, tmp_path):
        path = tmp_path / "part-0.parquet"
        sparse = {
            1: (np.array([2], dtype=np.int64), np.array([1.5], dtype=np.float32)),
            2: (np.array([], dtype=np.int64), np.array([], dtype=np.float32)),
        }

        write_sparse_vectors(path, sparse, 8, "desc")

        table = pq.read_table(path)
        assert table["indices"].to_pylist() == [[2], []]
        assert table["values"].to_pylist() == [[1.5], []]

    def test_every_row_empty_still_writes_a_valid_file(self, tmp_path):
        path = tmp_path / "part-0.parquet"
        sparse = {n: (np.array([], dtype=np.int64), np.array([], dtype=np.float32)) for n in (1, 2)}

        write_sparse_vectors(path, sparse, 4, "desc")

        assert pq.read_table(path)["indices"].to_pylist() == [[], []]

    def test_rejects_a_negative_index_with_a_named_error(self, tmp_path):
        path = tmp_path / "part-0.parquet"
        sparse = {7: (np.array([-1], dtype=np.int64), np.array([1.0], dtype=np.float32))}

        with pytest.raises(ValueError, match="node 7 has an index outside"):
            write_sparse_vectors(path, sparse, 4, "desc")

    def test_round_trips_ragged_rows_in_node_order_exactly(self, tmp_path):
        path = tmp_path / "part-0.parquet"
        sparse = {
            30: (np.array([0, 9], dtype=np.int64), np.array([1.0, 2.0], dtype=np.float32)),
            10: (np.array([4], dtype=np.int64), np.array([3.5], dtype=np.float32)),
            20: (np.array([], dtype=np.int64), np.array([], dtype=np.float32)),
        }

        write_sparse_vectors(path, sparse, 10, "desc")

        table = pq.read_table(path)
        assert table["node_id"].to_pylist() == [10, 20, 30]
        assert table["indices"].to_pylist() == [[4], [], [0, 9]]
        assert table["values"].to_pylist() == [[3.5], [], [1.0, 2.0]]

    def test_stores_indices_as_int32_and_values_as_float32(self, tmp_path):
        path = tmp_path / "part-0.parquet"
        sparse = {1: (np.array([3], dtype=np.int64), np.array([2.0], dtype=np.float64))}

        write_sparse_vectors(path, sparse, 8, "desc")

        schema = pq.read_schema(path)
        assert schema.field("indices").type.value_type == "int32"
        assert schema.field("values").type.value_type == "float"
