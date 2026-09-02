from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from core.export import dataset_path as _dataset_path
from core.support import build_signature_vocabulary
from syntax.corpus import PhrasePsalm
from syntax.scripts.generate_shuffle_control import (
    generate_shuffle_control,
    generate_signature_shuffle_control,
)


def dataset_path(output_root, vocab, weight):
    return _dataset_path(
        output_root, vocab, weight, domain="syntax", unit_key="feature", level="phrase"
    )


def _psalm(*, number, phrase_typ_by_half_verse, nodes, phrase_function_by_half_verse=None):
    return PhrasePsalm(
        number=number,
        half_verse_nodes=nodes,
        half_verse_typ=phrase_typ_by_half_verse,
        half_verse_function=phrase_function_by_half_verse
        or tuple(("Pred",) * len(c) for c in phrase_typ_by_half_verse),
    )


def _psalms():
    return [
        _psalm(
            number=1,
            phrase_typ_by_half_verse=(("NP", "VP", "PP"), ("VP", "NP", "CP")),
            nodes=(100, 101),
        ),
    ]


class TestGenerateShuffleControl:
    def test_writes_n_seeded_datasets(self, tmp_path):
        written = generate_shuffle_control(
            _psalms(), tmp_path, "typ", "1_2gram", n_shuffles=3, max_workers=1
        )

        assert written == [
            "typ_1_2gram_shuffle0001",
            "typ_1_2gram_shuffle0002",
            "typ_1_2gram_shuffle0003",
        ]
        for weight in written:
            assert dataset_path(tmp_path, "typ", weight.removeprefix("typ_")).exists()

    def test_each_shuffle_broadcasts_the_same_vector_to_every_half_verse_for_psalm_variants(
        self, tmp_path
    ):
        generate_shuffle_control(
            _psalms(), tmp_path, "typ", "1_2gram_psalm", n_shuffles=1, max_workers=1
        )

        table = pq.read_table(dataset_path(tmp_path, "typ", "1_2gram_psalm_shuffle0001"))
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert by_node[100] == by_node[101]

    def test_different_seeds_give_different_vectors(self, tmp_path):
        generate_shuffle_control(_psalms(), tmp_path, "typ", "1_2gram", n_shuffles=2, max_workers=1)

        table1 = pq.read_table(dataset_path(tmp_path, "typ", "1_2gram_shuffle0001"))
        table2 = pq.read_table(dataset_path(tmp_path, "typ", "1_2gram_shuffle0002"))
        assert table1["vector"].to_pylist() != table2["vector"].to_pylist()

    def test_the_trigram_family_is_written_sparsely(self, tmp_path):
        generate_shuffle_control(
            _psalms(), tmp_path, "typ", "1_2_3gram", n_shuffles=2, max_workers=1
        )

        table = pq.read_table(dataset_path(tmp_path, "typ", "1_2_3gram_shuffle0001"))

        assert table.schema.names == ["node_id", "indices", "values"]
        assert table.schema.metadata[b"sparse"] == b"true"

    def test_different_seeds_give_different_sparse_trigram_vectors(self, tmp_path):
        generate_shuffle_control(
            _psalms(), tmp_path, "typ", "1_2_3gram", n_shuffles=2, max_workers=1
        )

        first = pq.read_table(dataset_path(tmp_path, "typ", "1_2_3gram_shuffle0001"))
        second = pq.read_table(dataset_path(tmp_path, "typ", "1_2_3gram_shuffle0002"))

        assert first["indices"].to_pylist() != second["indices"].to_pylist()

    def test_raises_on_an_unshuffleable_representation(self, tmp_path):
        with pytest.raises(ValueError, match="no shuffle control"):
            generate_shuffle_control(
                _psalms(), tmp_path, "typ", "1gram", n_shuffles=1, max_workers=1
            )

    def test_writes_phrase_function_datasets_too(self, tmp_path):
        written = generate_shuffle_control(
            _psalms(), tmp_path, "function", "1_2gram", n_shuffles=2, max_workers=1
        )

        assert written == [
            "function_1_2gram_shuffle0001",
            "function_1_2gram_shuffle0002",
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
            max_workers=1,
            vocabulary=vocabulary,
            external_counts=_external_counts(),
            k=1000,
        )

        assert written == [
            "signature_1_2gram_shuffle0001",
            "signature_1_2gram_shuffle0002",
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
                max_workers=1,
                vocabulary=vocabulary,
                external_counts=_external_counts(),
                k=1000,
            )
