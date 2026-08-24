from __future__ import annotations

import pyarrow.parquet as pq

from lexical.export import dataset_path as _dataset_path
from syntax.corpus import PhrasePsalm
from syntax.scripts.generate_deploy_shuffle_control import generate_shuffle_control


def dataset_path(output_root, vocab, weight):
    return _dataset_path(
        output_root, vocab, weight, domain="syntax", unit_key="feature", level="phrase"
    )


def _psalm(*, number, nodes, typ, function):
    return PhrasePsalm(number=number, colon_nodes=nodes, colon_typ=typ, colon_function=function)


def _psalms():
    return [
        _psalm(
            number=1,
            nodes=(100, 101, 102),
            typ=(("NP",), ("VP",), ("PP",)),
            function=(("Subj",), ("Pred",), ("Cmpl",)),
        ),
    ]


def _vocab_and_counts():
    external_counts = {"NP:Subj": 5000, "VP:Pred": 5000, "PP:Cmpl": 5000}
    vocabulary = ("NP:Subj", "PP:Cmpl", "VP:Pred", "<RARE>")
    return vocabulary, external_counts


class TestGenerateShuffleControl:
    def test_writes_n_seeded_datasets(self, tmp_path):
        vocabulary, external_counts = _vocab_and_counts()
        written = generate_shuffle_control(
            _psalms(),
            tmp_path,
            n_shuffles=3,
            vocabulary=vocabulary,
            external_counts=external_counts,
            k=1000,
        )

        assert written == [
            "signature_posmean_shuffle01",
            "signature_posmean_shuffle02",
            "signature_posmean_shuffle03",
        ]
        for weight in ("posmean_shuffle01", "posmean_shuffle02", "posmean_shuffle03"):
            assert dataset_path(tmp_path, "signature", weight).exists()

    def test_each_shuffle_broadcasts_the_same_vector_to_every_colon(self, tmp_path):
        vocabulary, external_counts = _vocab_and_counts()
        generate_shuffle_control(
            _psalms(),
            tmp_path,
            n_shuffles=1,
            vocabulary=vocabulary,
            external_counts=external_counts,
            k=1000,
        )

        table = pq.read_table(dataset_path(tmp_path, "signature", "posmean_shuffle01"))
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert by_node[100] == by_node[101] == by_node[102]

    def test_different_seeds_give_different_vectors(self, tmp_path):
        vocabulary, external_counts = _vocab_and_counts()
        generate_shuffle_control(
            _psalms(),
            tmp_path,
            n_shuffles=2,
            vocabulary=vocabulary,
            external_counts=external_counts,
            k=1000,
        )

        table1 = pq.read_table(dataset_path(tmp_path, "signature", "posmean_shuffle01"))
        table2 = pq.read_table(dataset_path(tmp_path, "signature", "posmean_shuffle02"))
        vec1 = table1["vector"].to_pylist()[0]
        vec2 = table2["vector"].to_pylist()[0]
        assert vec1 != vec2
