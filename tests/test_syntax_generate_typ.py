from __future__ import annotations

import pyarrow.parquet as pq

from lexical.export import dataset_path as _dataset_path
from syntax.corpus import PhrasePsalm
from syntax.generate_typ import generate


def dataset_path(output_root, vocab, weight):
    return _dataset_path(
        output_root, vocab, weight, domain="syntax", unit_key="feature", level="phrase"
    )


_FULL_WEIGHTS = (
    "1gram",
    "1_2gram",
    "1_2_3gram",
    "1gram_psalm",
    "1_2gram_psalm",
    "1_2_3gram_psalm",
)


def _psalm(*, number, phrase_typ_by_colon, nodes):
    return PhrasePsalm(number=number, colon_nodes=nodes, colon_typ=phrase_typ_by_colon)


def _psalms():
    return [
        _psalm(number=1, phrase_typ_by_colon=(("NP", "VP"), ("PP",)), nodes=(100, 101)),
        _psalm(number=2, phrase_typ_by_colon=(("CP",),), nodes=(200,)),
    ]


class TestGenerate:
    def test_writes_the_full_phrase_typ_representation_family(self, tmp_path):
        written = generate(_psalms(), tmp_path)

        assert set(written) == {f"typ_{w}" for w in _FULL_WEIGHTS}
        for weight in _FULL_WEIGHTS:
            assert dataset_path(tmp_path, "typ", weight).exists()

    def test_skips_variants_whose_dataset_already_exists(self, tmp_path):
        generate(_psalms(), tmp_path)

        written_again = generate(_psalms(), tmp_path)

        assert written_again == []

    def test_psalm_variants_broadcast_the_same_vector_within_a_psalm(self, tmp_path):
        generate(_psalms(), tmp_path)

        table = pq.read_table(dataset_path(tmp_path, "typ", "1gram_psalm"))
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert by_node[100] == by_node[101]

    def test_colon_level_variants_give_distinct_colons_their_own_vector(self, tmp_path):
        generate(_psalms(), tmp_path)

        table = pq.read_table(dataset_path(tmp_path, "typ", "1gram"))
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert by_node[100] != by_node[101]
