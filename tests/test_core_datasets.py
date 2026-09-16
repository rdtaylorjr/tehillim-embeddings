"""Naming, finding, and reading datasets: the contract every consumer of the tree shares."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.datasets import (
    UnnamedDatasetError,
    dataset_identifier,
    dataset_paths,
    discover_domains,
    domain_root,
    is_sparse_embeddings,
    read_dense_rows,
    read_sparse_rows,
    sparse_rows_to_csr,
    split_model_name,
)
from core.export import write_sparse_vectors, write_vectors
from core.partition import BHSA_HALF_VERSE, Scope


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


BHSA = "corpus=bhsa/unit=half_verse"


class TestDatasetIdentifier:
    def test_reads_model_and_variation_from_the_hive_path(self) -> None:
        path = Path(f"data/{BHSA}/domain=semantic/model=bge_m3/text=vocalized/part-0.parquet")
        assert dataset_identifier(path) == "bge_m3_vocalized"

    def test_handles_a_lexical_path(self) -> None:
        path = Path(f"data/{BHSA}/domain=lexical/type=homograph/construction=binary/part-0.parquet")
        assert dataset_identifier(path) == "homograph_binary"

    def test_handles_a_lexical_path_with_a_text_tier(self) -> None:
        path = Path(
            f"data/{BHSA}/domain=lexical/type=word/text=consonantal/construction=binary/x.parquet"
        )
        assert dataset_identifier(path) == "word_consonantal_binary"

    def test_the_identifier_names_the_representation_not_the_scope(self) -> None:
        """Two corpora at one unit give a dataset the same identifier under their own scopes."""
        scroll = "corpus=dss/witness=11Q5/reconstruction=none/unit=verse"
        path = Path(f"data/{scroll}/domain=semantic/model=bge_m3/text=consonantal/part-0.parquet")
        assert dataset_identifier(path) == "bge_m3_consonantal"

    def test_refuses_a_file_outside_any_partition(self) -> None:
        with pytest.raises(UnnamedDatasetError, match="names no dataset"):
            dataset_identifier(Path("data/stray.parquet"))

    def test_refuses_a_partition_the_grammar_does_not_admit(self) -> None:
        with pytest.raises(UnnamedDatasetError, match="appears twice"):
            dataset_identifier(Path(f"data/{BHSA}/domain=lexical/unit=homograph/part-0.parquet"))


class TestSplitModelName:
    def test_extracts_base_and_variant(self) -> None:
        assert split_model_name("semantic_gemini_embedding_2_cantillation") == (
            "gemini_embedding_2",
            "cantillation",
        )
        assert split_model_name("semantic_bge_m3_vocalized") == ("bge_m3", "vocalized")

    def test_falls_back_to_unknown_variant(self) -> None:
        assert split_model_name("semantic_something_odd") == ("something_odd", "unknown")

    def test_extracts_a_variant_embedded_in_the_middle(self) -> None:
        assert split_model_name("word_consonantal_binary") == ("word_binary", "consonantal")

    def test_still_prefers_a_trailing_suffix_variant(self) -> None:
        assert split_model_name("bge_m3_vocalized") == ("bge_m3", "vocalized")


class TestDatasetPaths:
    def test_returns_every_dataset_file_in_canonical_order(self, tmp_path: Path) -> None:
        root = tmp_path / BHSA
        c = _touch(root / "domain=semantic/model=c/text=consonantal/part-0.parquet")
        a = _touch(root / "domain=semantic/model=a/text=consonantal/part-0.parquet")
        deep = _touch(
            root / "domain=lexical/type=word/text=vocalized/construction=icf/part-0.parquet"
        )
        assert dataset_paths(tmp_path) == [deep, a, c]

    def test_ignores_a_directory_that_merely_ends_in_parquet(self, tmp_path: Path) -> None:
        root = tmp_path / BHSA
        (root / "domain=semantic/stray.parquet").mkdir(parents=True)
        real = _touch(root / "domain=semantic/model=a/text=consonantal/part-0.parquet")
        assert dataset_paths(tmp_path) == [real]

    def test_returns_nothing_for_an_empty_directory(self, tmp_path: Path) -> None:
        assert dataset_paths(tmp_path) == []

    def test_excludes_order_shuffle_draws_of_either_generation(self, tmp_path: Path) -> None:
        """A shuffle draw is one sample of the order-shuffle null, not a dataset."""
        lexical = tmp_path / BHSA / "domain=lexical/type=homograph"
        real = _touch(lexical / "construction=icf_position4/part-0.parquet")
        _touch(lexical / "construction=icf_position4_shuffle0001/part-0.parquet")
        _touch(
            tmp_path
            / BHSA
            / "domain=morphological/feature=sp/construction=1_2gram_shuffle15/part-0.parquet"
        )
        assert dataset_paths(tmp_path) == [real]

    def test_keeps_a_dataset_whose_name_ends_in_the_bare_word_shuffle(self, tmp_path: Path) -> None:
        lexical = tmp_path / BHSA / "domain=lexical/type=homograph"
        kept = _touch(lexical / "construction=order_shuffle/part-0.parquet")
        assert dataset_paths(tmp_path) == [kept]

    def test_passes_over_a_parquet_outside_the_grammar(self, tmp_path: Path) -> None:
        _touch(tmp_path / "stray.parquet")
        _touch(tmp_path / BHSA / "domain=lexical/stray.parquet")
        _touch(tmp_path / BHSA / "domain=lexical/unit=homograph/construction=icf/vectors.parquet")
        real = _touch(tmp_path / BHSA / "domain=lexical/type=homograph/construction=icf/x.parquet")
        assert dataset_paths(tmp_path) == [real]


def test_discover_domains_lists_the_domain_directories_of_one_scope(tmp_path: Path) -> None:
    for domain in ("syntactic", "lexical"):
        (tmp_path / BHSA / f"domain={domain}").mkdir(parents=True)
    (tmp_path / BHSA / "domain=stray.txt").write_text("")
    (tmp_path / "corpus=dss/witness=11Q5/reconstruction=none/unit=verse/domain=semantic").mkdir(
        parents=True
    )
    assert discover_domains(tmp_path, BHSA_HALF_VERSE) == ("lexical", "syntactic")
    assert discover_domains(tmp_path, Scope("dss", "verse", "11Q5", "none")) == ("semantic",)


def test_domain_root_is_the_scope_root_and_the_domain(tmp_path: Path) -> None:
    assert domain_root(tmp_path, BHSA_HALF_VERSE, "lexical") == tmp_path / BHSA / "domain=lexical"


class TestReaders:
    def test_dense_rows_come_back_as_one_float32_matrix_with_zero_rows_kept(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / BHSA / "domain=lexical/type=lexeme/construction=count/d.parquet"
        write_vectors(path, {7: np.array([1.0, 2.0]), 3: np.array([0.0, 0.0])}, "d")
        rows = read_dense_rows(path, batch_size=1)
        assert list(rows) == [3, 7]
        assert rows[7].dtype == np.float32
        np.testing.assert_array_equal(rows[3], [0.0, 0.0])
        assert is_sparse_embeddings(path) is False

    def test_sparse_rows_keep_an_empty_row_at_its_place(self, tmp_path: Path) -> None:
        path = tmp_path / BHSA / "domain=lexical/type=lexeme/construction=count/s.parquet"
        empty = (np.array([], dtype="<i4"), np.array([], dtype="<f4"))
        write_sparse_vectors(
            path, {7: (np.array([2], dtype="<i4"), np.array([1.5], dtype="<f4")), 8: empty}, 4, "s"
        )
        node_ids, matrix = read_sparse_rows(path)
        assert node_ids == [7, 8]
        assert matrix.shape == (2, 4)
        assert matrix[0, 2] == pytest.approx(1.5)
        assert matrix[1].count_nonzero() == 0
        assert is_sparse_embeddings(path) is True

    def test_sparse_rows_to_csr_handles_all_empty_rows(self) -> None:
        node_ids, matrix = sparse_rows_to_csr([1, 2], [[], []], [[], []], 3)
        assert (node_ids, matrix.shape, matrix.nnz) == ([1, 2], (2, 3), 0)
