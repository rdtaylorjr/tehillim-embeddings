from __future__ import annotations

import pyarrow.parquet as pq

from lexical.corpus import LexicalPsalm
from lexical.export import dataset_path
from lexical.scripts.generate_shuffle_control_colon import generate_shuffle_control_colon


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
            number=1,
            lexemes=(("A", "B"), ("A",), ("B",), ("A", "B")),
            forms=(("A0", "B0"), ("A0",), ("B0",), ("A0", "B0")),
            nodes=(100, 101, 102, 103),
        ),
    ]


def _icf_weights():
    return {"A0": 1.5, "B0": 2.0}


class TestGenerateShuffleControlColon:
    def test_writes_n_seeded_datasets(self, tmp_path):
        written = generate_shuffle_control_colon(_psalms(), tmp_path, _icf_weights(), n_shuffles=3)

        assert written == [
            "icf_pos4_shuffle01",
            "icf_pos4_shuffle02",
            "icf_pos4_shuffle03",
        ]
        for weight in written:
            assert dataset_path(tmp_path, "lex0", weight).exists()

    def test_each_shuffle_gives_distinct_colons_distinct_vectors(self, tmp_path):
        generate_shuffle_control_colon(_psalms(), tmp_path, _icf_weights(), n_shuffles=1)

        table = pq.read_table(dataset_path(tmp_path, "lex0", "icf_pos4_shuffle01"))
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert by_node[100] != by_node[101]

    def test_different_seeds_give_different_vectors(self, tmp_path):
        generate_shuffle_control_colon(_psalms(), tmp_path, _icf_weights(), n_shuffles=2)

        table1 = pq.read_table(dataset_path(tmp_path, "lex0", "icf_pos4_shuffle01"))
        table2 = pq.read_table(dataset_path(tmp_path, "lex0", "icf_pos4_shuffle02"))
        assert table1["vector"].to_pylist() != table2["vector"].to_pylist()
