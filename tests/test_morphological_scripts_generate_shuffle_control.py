from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from lexical.export import dataset_path as _dataset_path
from morphological.corpus import MorphologicalPsalm
from morphological.scripts.generate_shuffle_control import (
    generate_shuffle_control,
    generate_signature_shuffle_control,
)


def dataset_path(output_root, vocab, weight):
    return _dataset_path(output_root, vocab, weight, dataset_type="morphological")


def _psalm(*, number, sp_by_colon, nodes, **feature_columns):
    return MorphologicalPsalm(
        number=number,
        half_verse_nodes=nodes,
        half_verse_sp=sp_by_colon,
        **{f"half_verse_{feature}": values for feature, values in feature_columns.items()},
    )


def _psalms():
    return [
        _psalm(
            number=1,
            sp_by_colon=(("subs", "verb", "prep"), ("verb", "subs", "conj")),
            nodes=(100, 101),
        ),
    ]


def _signature_psalms():
    all_na = (("NA", "NA", "NA"), ("NA", "NA", "NA"))
    return [
        _psalm(
            number=1,
            sp_by_colon=(("subs", "verb", "prep"), ("verb", "subs", "conj")),
            nodes=(200, 201),
            gn=all_na,
            nu=all_na,
            ps=all_na,
            st=all_na,
            vs=all_na,
            vt=all_na,
            prs_gn=all_na,
            prs_nu=all_na,
            prs_ps=all_na,
        ),
    ]


def _external_counts():
    return {"subs": 5000, "verb": 5000, "prep": 5000, "conj": 5000}


class TestGenerateShuffleControl:
    def test_writes_n_seeded_datasets_for_a_colon_level_representation(self, tmp_path):
        written = generate_shuffle_control(
            _psalms(), tmp_path, representation="1_2gram", n_shuffles=3
        )

        assert written == [
            "sp_1_2gram_shuffle01",
            "sp_1_2gram_shuffle02",
            "sp_1_2gram_shuffle03",
        ]
        for weight in ("1_2gram_shuffle01", "1_2gram_shuffle02", "1_2gram_shuffle03"):
            assert dataset_path(tmp_path, "sp", weight).exists()

    def test_writes_n_seeded_datasets_for_a_psalm_broadcast_representation(self, tmp_path):
        written = generate_shuffle_control(
            _psalms(), tmp_path, representation="1_2_3gram_psalm", n_shuffles=2
        )

        assert written == [
            "sp_1_2_3gram_psalm_shuffle01",
            "sp_1_2_3gram_psalm_shuffle02",
        ]
        assert dataset_path(tmp_path, "sp", "1_2_3gram_psalm_shuffle01").exists()

    def test_psalm_broadcast_variants_still_broadcast_within_the_shuffled_psalm(self, tmp_path):
        generate_shuffle_control(_psalms(), tmp_path, representation="1_2gram_psalm", n_shuffles=1)

        table = pq.read_table(dataset_path(tmp_path, "sp", "1_2gram_psalm_shuffle01"))
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert by_node[100] == by_node[101]

    def test_different_seeds_give_different_vectors(self, tmp_path):
        generate_shuffle_control(_psalms(), tmp_path, representation="1_2gram", n_shuffles=2)

        table1 = pq.read_table(dataset_path(tmp_path, "sp", "1_2gram_shuffle01"))
        table2 = pq.read_table(dataset_path(tmp_path, "sp", "1_2gram_shuffle02"))
        assert table1["vector"].to_pylist() != table2["vector"].to_pylist()

    def test_rejects_the_unigram_representation_which_has_no_shuffle_control(self, tmp_path):
        with pytest.raises(ValueError, match="unigram"):
            generate_shuffle_control(_psalms(), tmp_path, representation="unigram", n_shuffles=1)


class TestGenerateSignatureShuffleControl:
    def test_writes_n_seeded_morph_signature_datasets(self, tmp_path):
        vocabulary = ("conj", "prep", "subs", "verb", "<RARE>")
        written = generate_signature_shuffle_control(
            _signature_psalms(),
            tmp_path,
            representation="1_2gram",
            n_shuffles=2,
            vocabulary=vocabulary,
            external_counts=_external_counts(),
            k=1000,
        )

        assert written == [
            "morph_signature_1_2gram_shuffle01",
            "morph_signature_1_2gram_shuffle02",
        ]
        assert dataset_path(tmp_path, "morph_signature", "1_2gram_shuffle01").exists()

    def test_rejects_a_representation_with_no_shuffle_control(self, tmp_path):
        with pytest.raises(ValueError, match="inventory"):
            generate_signature_shuffle_control(
                _signature_psalms(),
                tmp_path,
                representation="inventory",
                n_shuffles=1,
                vocabulary=(),
                external_counts={},
                k=1000,
            )
