from __future__ import annotations

import pyarrow.parquet as pq

from lexical.corpus import LexicalPsalm
from lexical.export import dataset_path
from lexical.scripts.generate_shuffle_control import generate_shuffle_control


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
            lexemes=(("A", "B"), ("A",), ("B",)),
            forms=(("A0", "B0"), ("A0",), ("B0",)),
            nodes=(100, 101, 102),
        ),
    ]


def _icf_weights():
    return {"A0": 1.5, "B0": 2.0}


class TestGenerateShuffleControl:
    def test_writes_n_seeded_datasets(self, tmp_path):
        written = generate_shuffle_control(_psalms(), tmp_path, _icf_weights(), n_shuffles=3)

        assert written == [
            "icf_position_mean_psalm_shuffle01",
            "icf_position_mean_psalm_shuffle02",
            "icf_position_mean_psalm_shuffle03",
        ]
        for weight in written:
            assert dataset_path(tmp_path, "homograph", weight, unit_key="unit").exists()

    def test_each_shuffle_broadcasts_the_same_vector_to_every_colon(self, tmp_path):
        generate_shuffle_control(_psalms(), tmp_path, _icf_weights(), n_shuffles=1)

        table = pq.read_table(
            dataset_path(
                tmp_path, "homograph", "icf_position_mean_psalm_shuffle01", unit_key="unit"
            )
        )
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert by_node[100] == by_node[101] == by_node[102]

    def test_different_seeds_give_different_vectors(self, tmp_path):
        generate_shuffle_control(_psalms(), tmp_path, _icf_weights(), n_shuffles=2)

        table1 = pq.read_table(
            dataset_path(
                tmp_path, "homograph", "icf_position_mean_psalm_shuffle01", unit_key="unit"
            )
        )
        table2 = pq.read_table(
            dataset_path(
                tmp_path, "homograph", "icf_position_mean_psalm_shuffle02", unit_key="unit"
            )
        )
        vec1 = table1["vector"].to_pylist()[0]
        vec2 = table2["vector"].to_pylist()[0]
        assert vec1 != vec2
