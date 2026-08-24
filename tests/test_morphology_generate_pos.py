from __future__ import annotations

import pyarrow.parquet as pq

from lexical.export import dataset_path as _dataset_path
from morphology.corpus import MorphologicalPsalm
from morphology.generate_pos import generate


def dataset_path(output_root, vocab, weight):
    return _dataset_path(output_root, vocab, weight, domain="morphology", unit_key="feature")


_FULL_WEIGHTS = (
    "unigram",
    "1_2gram",
    "1_2_3gram",
    "unigram_psalm",
    "1_2gram_psalm",
    "1_2_3gram_psalm",
)


def _psalm(*, number, sp_by_colon, nodes):
    return MorphologicalPsalm(number=number, colon_nodes=nodes, colon_sp=sp_by_colon)


def _psalms():
    return [
        _psalm(number=1, sp_by_colon=(("subs", "verb"), ("subs",)), nodes=(100, 101)),
        _psalm(number=2, sp_by_colon=(("prep",),), nodes=(200,)),
    ]


class TestGenerate:
    def test_writes_the_full_pos_representation_family(self, tmp_path):
        written = generate(_psalms(), tmp_path)

        assert set(written) == {f"sp_{w}" for w in _FULL_WEIGHTS}
        for weight in _FULL_WEIGHTS:
            assert dataset_path(tmp_path, "sp", weight).exists()

    def test_skips_variants_whose_dataset_already_exists(self, tmp_path):
        generate(_psalms(), tmp_path)

        written_again = generate(_psalms(), tmp_path)

        assert written_again == []

    def test_identifier_reads_cleanly_as_unit_plus_construction(self, tmp_path):
        # feature=sp, construction=unigram -> "sp_unigram", not a doubled "sp_sp_unigram".
        generate(_psalms(), tmp_path)

        path = dataset_path(tmp_path, "sp", "unigram")
        assert path.parent.name == "construction=unigram"
        assert path.parent.parent.name == "feature=sp"

    def test_psalm_variants_broadcast_the_same_vector_within_a_psalm(self, tmp_path):
        generate(_psalms(), tmp_path)

        table = pq.read_table(dataset_path(tmp_path, "sp", "unigram_psalm"))
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert by_node[100] == by_node[101]

    def test_colon_level_variants_give_distinct_colons_their_own_vector(self, tmp_path):
        generate(_psalms(), tmp_path)

        table = pq.read_table(dataset_path(tmp_path, "sp", "unigram"))
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert by_node[100] != by_node[101]
