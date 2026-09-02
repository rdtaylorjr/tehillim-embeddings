from __future__ import annotations

import pyarrow.parquet as pq

from core.export import dataset_path as _dataset_path
from morphology.corpus import MorphologicalPsalm
from morphology.scripts.generate_deploy_shuffle_control import generate_shuffle_control


def dataset_path(output_root, vocab, weight):
    return _dataset_path(output_root, vocab, weight, domain="morphology", unit_key="feature")


def _psalm(*, number, nodes, **feature_columns):
    return MorphologicalPsalm(
        number=number,
        half_verse_nodes=nodes,
        **{f"half_verse_{feature}": values for feature, values in feature_columns.items()},
    )


def _psalms():
    return [
        _psalm(
            number=1,
            nodes=(100, 101, 102),
            prs_gn=(("NA",), ("m",), ("NA",)),
            prs_nu=(("NA",), ("pl",), ("NA",)),
            prs_ps=(("NA",), ("p3",), ("NA",)),
        ),
    ]


class TestGenerateShuffleControl:
    def test_writes_n_seeded_datasets(self, tmp_path):
        written = generate_shuffle_control(_psalms(), tmp_path, n_shuffles=3, max_workers=1)

        assert written == [
            "morph_suffix_posmean_shuffle0001",
            "morph_suffix_posmean_shuffle0002",
            "morph_suffix_posmean_shuffle0003",
        ]
        for weight in ("posmean_shuffle0001", "posmean_shuffle0002", "posmean_shuffle0003"):
            assert dataset_path(tmp_path, "morph_suffix", weight).exists()

    def test_each_shuffle_broadcasts_the_same_vector_to_every_half_verse(self, tmp_path):
        generate_shuffle_control(_psalms(), tmp_path, n_shuffles=1, max_workers=1)

        table = pq.read_table(dataset_path(tmp_path, "morph_suffix", "posmean_shuffle0001"))
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert by_node[100] == by_node[101] == by_node[102]

    def test_different_seeds_give_different_vectors(self, tmp_path):
        generate_shuffle_control(_psalms(), tmp_path, n_shuffles=2, max_workers=1)

        table1 = pq.read_table(dataset_path(tmp_path, "morph_suffix", "posmean_shuffle0001"))
        table2 = pq.read_table(dataset_path(tmp_path, "morph_suffix", "posmean_shuffle0002"))
        vec1 = table1["vector"].to_pylist()[0]
        vec2 = table2["vector"].to_pylist()[0]
        assert vec1 != vec2
