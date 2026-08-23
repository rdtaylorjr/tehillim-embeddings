from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from lexical.export import dataset_path as _dataset_path
from phrase.corpus import PhrasePsalm
from phrase.scripts.generate_shuffle_control import generate_shuffle_control


def dataset_path(output_root, vocab, weight):
    return _dataset_path(output_root, vocab, weight, dataset_type="phrase")


def _psalm(*, number, typ_by_colon, nodes, function_by_colon=None):
    return PhrasePsalm(
        number=number,
        half_verse_nodes=nodes,
        half_verse_typ=typ_by_colon,
        half_verse_function=function_by_colon or tuple(("Pred",) * len(c) for c in typ_by_colon),
    )


def _psalms():
    return [
        _psalm(
            number=1,
            typ_by_colon=(("NP", "VP", "PP"), ("VP", "NP", "CP")),
            nodes=(100, 101),
        ),
    ]


class TestGenerateShuffleControl:
    def test_writes_n_seeded_datasets(self, tmp_path):
        written = generate_shuffle_control(
            _psalms(), tmp_path, "phrase_typ", "1_2gram", n_shuffles=3
        )

        assert written == [
            "phrase_typ_1_2gram_shuffle01",
            "phrase_typ_1_2gram_shuffle02",
            "phrase_typ_1_2gram_shuffle03",
        ]
        for weight in written:
            assert dataset_path(tmp_path, "phrase_typ", weight.removeprefix("phrase_typ_")).exists()

    def test_each_shuffle_broadcasts_the_same_vector_to_every_colon_for_psalm_variants(
        self, tmp_path
    ):
        generate_shuffle_control(_psalms(), tmp_path, "phrase_typ", "1_2gram_psalm", n_shuffles=1)

        table = pq.read_table(dataset_path(tmp_path, "phrase_typ", "1_2gram_psalm_shuffle01"))
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert by_node[100] == by_node[101]

    def test_different_seeds_give_different_vectors(self, tmp_path):
        generate_shuffle_control(_psalms(), tmp_path, "phrase_typ", "1_2_3gram", n_shuffles=2)

        table1 = pq.read_table(dataset_path(tmp_path, "phrase_typ", "1_2_3gram_shuffle01"))
        table2 = pq.read_table(dataset_path(tmp_path, "phrase_typ", "1_2_3gram_shuffle02"))
        assert table1["vector"].to_pylist() != table2["vector"].to_pylist()

    def test_raises_on_an_unshuffleable_representation(self, tmp_path):
        with pytest.raises(ValueError, match="no shuffle control"):
            generate_shuffle_control(_psalms(), tmp_path, "phrase_typ", "1gram", n_shuffles=1)

    def test_writes_phrase_function_datasets_too(self, tmp_path):
        written = generate_shuffle_control(
            _psalms(), tmp_path, "phrase_function", "1_2gram", n_shuffles=2
        )

        assert written == [
            "phrase_function_1_2gram_shuffle01",
            "phrase_function_1_2gram_shuffle02",
        ]
        for weight in written:
            path_weight = weight.removeprefix("phrase_function_")
            assert dataset_path(tmp_path, "phrase_function", path_weight).exists()
