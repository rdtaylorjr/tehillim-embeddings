from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from lexical.export import dataset_path as _dataset_path
from syntax.corpus import PhrasePsalm
from syntax.scripts.generate_shuffle_control import (
    generate_shuffle_control,
    generate_signature_shuffle_control,
)
from syntax.signature_support import build_signature_vocabulary


def dataset_path(output_root, vocab, weight):
    return _dataset_path(
        output_root, vocab, weight, domain="syntax", unit_key="feature", level="phrase"
    )


def _psalm(*, number, phrase_typ_by_colon, nodes, phrase_function_by_colon=None):
    return PhrasePsalm(
        number=number,
        half_verse_nodes=nodes,
        half_verse_typ=phrase_typ_by_colon,
        half_verse_function=phrase_function_by_colon
        or tuple(("Pred",) * len(c) for c in phrase_typ_by_colon),
    )


def _psalms():
    return [
        _psalm(
            number=1,
            phrase_typ_by_colon=(("NP", "VP", "PP"), ("VP", "NP", "CP")),
            nodes=(100, 101),
        ),
    ]


class TestGenerateShuffleControl:
    def test_writes_n_seeded_datasets(self, tmp_path):
        written = generate_shuffle_control(_psalms(), tmp_path, "typ", "1_2gram", n_shuffles=3)

        assert written == [
            "typ_1_2gram_shuffle01",
            "typ_1_2gram_shuffle02",
            "typ_1_2gram_shuffle03",
        ]
        for weight in written:
            assert dataset_path(tmp_path, "typ", weight.removeprefix("typ_")).exists()

    def test_each_shuffle_broadcasts_the_same_vector_to_every_colon_for_psalm_variants(
        self, tmp_path
    ):
        generate_shuffle_control(_psalms(), tmp_path, "typ", "1_2gram_psalm", n_shuffles=1)

        table = pq.read_table(dataset_path(tmp_path, "typ", "1_2gram_psalm_shuffle01"))
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert by_node[100] == by_node[101]

    def test_different_seeds_give_different_vectors(self, tmp_path):
        generate_shuffle_control(_psalms(), tmp_path, "typ", "1_2_3gram", n_shuffles=2)

        table1 = pq.read_table(dataset_path(tmp_path, "typ", "1_2_3gram_shuffle01"))
        table2 = pq.read_table(dataset_path(tmp_path, "typ", "1_2_3gram_shuffle02"))
        assert table1["vector"].to_pylist() != table2["vector"].to_pylist()

    def test_raises_on_an_unshuffleable_representation(self, tmp_path):
        with pytest.raises(ValueError, match="no shuffle control"):
            generate_shuffle_control(_psalms(), tmp_path, "typ", "1gram", n_shuffles=1)

    def test_writes_phrase_function_datasets_too(self, tmp_path):
        written = generate_shuffle_control(_psalms(), tmp_path, "function", "1_2gram", n_shuffles=2)

        assert written == [
            "function_1_2gram_shuffle01",
            "function_1_2gram_shuffle02",
        ]
        for weight in written:
            path_weight = weight.removeprefix("function_")
            assert dataset_path(tmp_path, "function", path_weight).exists()


def _external_counts():
    return {"NP:Subj": 5000, "VP:Pred": 5000}


class TestGenerateSignatureShuffleControl:
    def test_writes_n_seeded_signature_datasets(self, tmp_path):
        vocabulary = build_signature_vocabulary(_external_counts(), k=1000)

        written = generate_signature_shuffle_control(
            _psalms(),
            tmp_path,
            "1_2gram",
            n_shuffles=2,
            vocabulary=vocabulary,
            external_counts=_external_counts(),
            k=1000,
        )

        assert written == [
            "signature_1_2gram_shuffle01",
            "signature_1_2gram_shuffle02",
        ]
        for weight in written:
            path_weight = weight.removeprefix("signature_")
            assert dataset_path(tmp_path, "signature", path_weight).exists()

    def test_raises_on_an_unshuffleable_representation(self, tmp_path):
        vocabulary = build_signature_vocabulary(_external_counts(), k=1000)

        with pytest.raises(ValueError, match="no shuffle control"):
            generate_signature_shuffle_control(
                _psalms(),
                tmp_path,
                "inventory",
                n_shuffles=1,
                vocabulary=vocabulary,
                external_counts=_external_counts(),
                k=1000,
            )
