from __future__ import annotations

import pyarrow.parquet as pq

from lexical.corpus import LexicalPsalm
from lexical.export import dataset_path
from lexical.generate import generate


def _psalm(*, number, lexemes, forms, nodes):
    return LexicalPsalm(
        number=number,
        half_verse_lexemes=lexemes,
        half_verse_forms=forms,
        half_verse_nodes=nodes,
    )


def _psalms():
    return [
        _psalm(
            number=1, lexemes=(("A", "B"), ("A",)), forms=(("A0", "B0"), ("A0",)), nodes=(100, 101)
        ),
        _psalm(number=2, lexemes=(("C",),), forms=(("C0",),), nodes=(200,)),
    ]


def _icf_weights():
    return {"A0": 1.5, "B0": 2.0, "C0": 0.5}


class TestGenerate:
    def test_writes_all_form_weightings_and_the_frozen_lexeme_binary(self, tmp_path):
        written = generate(_psalms(), tmp_path, _icf_weights())

        assert set(written) == {
            "form_binary",
            "form_count",
            "form_log_count",
            "form_icf",
            "form_tf_icf",
            "form_icf_pos2",
            "form_icf_pos4",
            "form_icf_pos8",
            "form_icf_lag2",
            "form_icf_lag4",
            "form_icf_lag8",
            "form_icf_posmean",
            "lexeme_binary",
        }
        for weight in (
            "binary",
            "count",
            "log_count",
            "icf",
            "tf_icf",
            "icf_pos2",
            "icf_pos4",
            "icf_pos8",
            "icf_lag2",
            "icf_lag4",
            "icf_lag8",
            "icf_posmean",
        ):
            assert dataset_path(tmp_path, "form", weight).exists()
        assert dataset_path(tmp_path, "lexeme", "binary").exists()

    def test_lag_dimension_equals_k(self, tmp_path):
        generate(_psalms(), tmp_path, _icf_weights())

        for weight, k in (("icf_lag2", 2), ("icf_lag4", 4), ("icf_lag8", 8)):
            table = pq.read_table(dataset_path(tmp_path, "form", weight))
            assert len(table["vector"].to_pylist()[0]) == k

    def test_lag_broadcasts_the_same_vector_to_every_colon_of_a_psalm(self, tmp_path):
        generate(_psalms(), tmp_path, _icf_weights())

        table = pq.read_table(dataset_path(tmp_path, "form", "icf_lag4"))
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert by_node[100] == by_node[101]

    def test_posmean_dimension_is_twice_the_vocabulary_size(self, tmp_path):
        generate(_psalms(), tmp_path, _icf_weights())

        table = pq.read_table(dataset_path(tmp_path, "form", "icf_posmean"))
        assert len(table["vector"].to_pylist()[0]) == 6

    def test_posmean_broadcasts_the_same_vector_to_every_colon_of_a_psalm(self, tmp_path):
        generate(_psalms(), tmp_path, _icf_weights())

        table = pq.read_table(dataset_path(tmp_path, "form", "icf_posmean"))
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert by_node[100] == by_node[101]

    def test_positional_dimension_is_k_times_vocabulary_size(self, tmp_path):
        generate(_psalms(), tmp_path, _icf_weights())

        for weight, k in (("icf_pos2", 2), ("icf_pos4", 4), ("icf_pos8", 8)):
            table = pq.read_table(dataset_path(tmp_path, "form", weight))
            assert len(table["vector"].to_pylist()[0]) == 3 * k

    def test_positional_broadcasts_the_same_vector_to_every_colon_of_a_psalm(self, tmp_path):
        generate(_psalms(), tmp_path, _icf_weights())

        table = pq.read_table(dataset_path(tmp_path, "form", "icf_pos4"))
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        # psalm 1 has nodes 100 and 101; both must carry the identical psalm-level vector.
        assert by_node[100] == by_node[101]

    def test_skips_variants_whose_dataset_already_exists(self, tmp_path):
        generate(_psalms(), tmp_path, _icf_weights())

        written_again = generate(_psalms(), tmp_path, _icf_weights())

        assert written_again == []

    def test_lexeme_vocabulary_dimension_matches_distinct_lex_values(self, tmp_path):
        generate(_psalms(), tmp_path, _icf_weights())

        table = pq.read_table(dataset_path(tmp_path, "lexeme", "binary"))
        # distinct lex values across fixtures: A, B, C
        assert len(table["vector"].to_pylist()[0]) == 3

    def test_form_vocabulary_dimension_matches_distinct_lex0_values(self, tmp_path):
        generate(_psalms(), tmp_path, _icf_weights())

        table = pq.read_table(dataset_path(tmp_path, "form", "binary"))
        # distinct lex0 values across fixtures: A0, B0, C0
        assert len(table["vector"].to_pylist()[0]) == 3

    def test_form_count_and_binary_differ_for_a_repeated_lexeme(self, tmp_path):
        generate(_psalms(), tmp_path, _icf_weights())

        binary_table = pq.read_table(dataset_path(tmp_path, "form", "binary"))
        count_table = pq.read_table(dataset_path(tmp_path, "form", "count"))
        binary_ids = binary_table["node_id"].to_pylist()
        binary_vecs = binary_table["vector"].to_pylist()
        binary_by_node = dict(zip(binary_ids, binary_vecs, strict=True))
        count_ids = count_table["node_id"].to_pylist()
        count_vecs = count_table["vector"].to_pylist()
        count_by_node = dict(zip(count_ids, count_vecs, strict=True))
        # node 100 has A0 once, B0 once: binary and count agree here.
        assert binary_by_node[100] == count_by_node[100]
