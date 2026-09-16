"""The partition writers: dense and sparse files land at the partition's path with its keys."""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq

from core.export import write_dataset, write_sparse_dataset
from core.partition import BHSA_HALF_VERSE, Partition

HOMOGRAPH_BINARY = Partition(BHSA_HALF_VERSE, "lexical", type="homograph", construction="binary")
WORD_BINARY = Partition(
    BHSA_HALF_VERSE, "lexical", type="word", text="consonantal", construction="binary"
)
SP_UNIGRAM = Partition(BHSA_HALF_VERSE, "morphological", feature="sp", construction="sp_unigram")
SIGNATURE_TRIGRAM = Partition(
    BHSA_HALF_VERSE, "morphological", feature="morph_signature", construction="1_2_3gram"
)


class TestWriteDataset:
    def test_round_trips_node_ids_and_vectors_exactly(self, tmp_path):
        vectors = {
            100: np.array([1.0, 0.0, 1.0], dtype=np.float32),
            101: np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }

        write_dataset(tmp_path, HOMOGRAPH_BINARY, vectors, "a description")

        table = pq.read_table(HOMOGRAPH_BINARY.file(tmp_path))
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert set(by_node) == {100, 101}
        assert np.array_equal(by_node[100], vectors[100])
        assert np.array_equal(by_node[101], vectors[101])

    def test_stores_float32_regardless_of_input_dtype(self, tmp_path):
        vectors = {100: np.array([1.0, 0.0], dtype=np.float64)}

        write_dataset(tmp_path, HOMOGRAPH_BINARY, vectors, "a description")

        table = pq.read_table(HOMOGRAPH_BINARY.file(tmp_path))
        assert str(table["vector"].type.value_type) == "float"

    def test_stores_the_description_and_the_partition_keys_in_file_metadata(self, tmp_path):
        vectors = {100: np.array([1.0], dtype=np.float32)}

        write_dataset(tmp_path, HOMOGRAPH_BINARY, vectors, "a specific description")

        metadata = pq.read_table(HOMOGRAPH_BINARY.file(tmp_path)).schema.metadata
        assert metadata[b"description"] == b"a specific description"
        assert metadata[b"corpus"] == b"bhsa"
        assert metadata[b"unit"] == b"half_verse"
        assert metadata[b"domain"] == b"lexical"
        assert metadata[b"type"] == b"homograph"
        assert metadata[b"construction"] == b"binary"
        assert b"witness" not in metadata

    def test_creates_parent_directories(self, tmp_path):
        vectors = {100: np.array([1.0], dtype=np.float32)}

        write_dataset(tmp_path / "nested", HOMOGRAPH_BINARY, vectors, "d")

        assert HOMOGRAPH_BINARY.file(tmp_path / "nested").exists()

    def test_writes_to_the_text_partitioned_path_of_a_surface_type(self, tmp_path):
        vectors = {100: np.array([1.0], dtype=np.float32)}

        write_dataset(tmp_path, WORD_BINARY, vectors, "d")

        assert WORD_BINARY.file(tmp_path).exists()
        assert pq.read_table(WORD_BINARY.file(tmp_path)).schema.metadata[b"text"] == b"consonantal"

    def test_writes_under_another_domain(self, tmp_path):
        vectors = {100: np.array([1.0], dtype=np.float32)}

        write_dataset(tmp_path, SP_UNIGRAM, vectors, "d")

        assert SP_UNIGRAM.file(tmp_path).exists()


class TestWriteSparseDataset:
    def test_round_trips_indices_and_values_exactly(self, tmp_path):
        sparse_vectors = {
            100: (np.array([2, 5000], dtype=np.int32), np.array([1.0, 0.5], dtype=np.float32)),
            101: (np.array([9], dtype=np.int32), np.array([2.0], dtype=np.float32)),
        }

        write_sparse_dataset(
            tmp_path, SIGNATURE_TRIGRAM, sparse_vectors, dim=74088, description="d"
        )

        table = pq.read_table(SIGNATURE_TRIGRAM.file(tmp_path))
        by_node = {
            node: (idx, val)
            for node, idx, val in zip(
                table["node_id"].to_pylist(),
                table["indices"].to_pylist(),
                table["values"].to_pylist(),
                strict=True,
            )
        }
        assert by_node[100] == ([2, 5000], [1.0, 0.5])
        assert by_node[101] == ([9], [2.0])

    def test_stores_the_dimension_the_sparse_flag_and_the_partition_keys(self, tmp_path):
        sparse_vectors = {100: (np.array([0], dtype=np.int32), np.array([1.0], dtype=np.float32))}

        write_sparse_dataset(
            tmp_path, SIGNATURE_TRIGRAM, sparse_vectors, dim=74088, description="d"
        )

        metadata = pq.read_table(SIGNATURE_TRIGRAM.file(tmp_path)).schema.metadata
        assert metadata[b"dim"] == b"74088"
        assert metadata[b"sparse"] == b"true"
        assert metadata[b"feature"] == b"morph_signature"

    def test_handles_an_empty_sparse_vector(self, tmp_path):
        sparse_vectors = {100: (np.array([], dtype=np.int32), np.array([], dtype=np.float32))}

        write_sparse_dataset(
            tmp_path, SIGNATURE_TRIGRAM, sparse_vectors, dim=74088, description="d"
        )

        table = pq.read_table(SIGNATURE_TRIGRAM.file(tmp_path))
        assert table["indices"].to_pylist() == [[]]
        assert table["values"].to_pylist() == [[]]

    def test_creates_parent_directories(self, tmp_path):
        sparse_vectors = {100: (np.array([0], dtype=np.int32), np.array([1.0], dtype=np.float32))}

        write_sparse_dataset(
            tmp_path / "nested", SIGNATURE_TRIGRAM, sparse_vectors, dim=74088, description="d"
        )

        assert SIGNATURE_TRIGRAM.file(tmp_path / "nested").exists()
