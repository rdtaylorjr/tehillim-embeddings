from __future__ import annotations

import pyarrow.parquet as pq

from core.export import dataset_path
from core.shuffle import shuffle_construction_name
from lexical.corpus import LexicalPsalm
from lexical.scripts.generate_shuffle_control import _CONSTRUCTION, build_vectors
from lexical.scripts.shuffle_driver import generate


def _psalms():
    return [
        LexicalPsalm(
            number=1,
            half_verse_lexemes=(("A", "B"), ("A",), ("B",)),
            half_verse_forms=(("A0", "B0"), ("A0",), ("B0",)),
            half_verse_nodes=(100, 101, 102),
        ),
    ]


def _icf_weights():
    return {"A0": 1.5, "B0": 2.0}


def _generate(tmp_path, n_shuffles):
    return generate(
        _psalms(),
        tmp_path,
        _icf_weights(),
        n_shuffles,
        construction=_CONSTRUCTION,
        builder=build_vectors,
        max_workers=1,
    )


def _table(tmp_path, seed):
    name = shuffle_construction_name(_CONSTRUCTION, seed)
    return pq.read_table(dataset_path(tmp_path, "homograph", name, unit_key="unit"))


class TestGenerateShuffleControl:
    def test_writes_n_seeded_datasets(self, tmp_path):
        written = _generate(tmp_path, 3)

        assert written == [shuffle_construction_name(_CONSTRUCTION, seed) for seed in (1, 2, 3)]
        for name in written:
            assert dataset_path(tmp_path, "homograph", name, unit_key="unit").exists()

    def test_each_shuffle_broadcasts_the_same_vector_to_every_half_verse(self, tmp_path):
        _generate(tmp_path, 1)

        table = _table(tmp_path, 1)
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert by_node[100] == by_node[101] == by_node[102]

    def test_different_seeds_give_different_vectors(self, tmp_path):
        _generate(tmp_path, 2)

        assert (
            _table(tmp_path, 1)["vector"].to_pylist()[0]
            != _table(tmp_path, 2)["vector"].to_pylist()[0]
        )
