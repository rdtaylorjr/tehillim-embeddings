from __future__ import annotations

import pyarrow.parquet as pq

from lexical.export import dataset_path as _dataset_path
from phrase.corpus import PhrasePsalm
from phrase.generate_det import generate


def dataset_path(output_root, vocab, weight):
    return _dataset_path(output_root, vocab, weight, dataset_type="phrase")


def _psalm(*, number, nodes, det):
    return PhrasePsalm(number=number, half_verse_nodes=nodes, half_verse_det=det)


def _psalms():
    return [_psalm(number=1, nodes=(100, 101), det=(("det", "und"), ("NA",)))]


class TestGenerate:
    def test_writes_both_constructions(self, tmp_path):
        written = generate(_psalms(), tmp_path)

        assert set(written) == {"phrase_det_1gram", "phrase_det_1gram_psalm"}
        assert dataset_path(tmp_path, "phrase_det", "1gram").exists()
        assert dataset_path(tmp_path, "phrase_det", "1gram_psalm").exists()

    def test_skips_variants_whose_dataset_already_exists(self, tmp_path):
        generate(_psalms(), tmp_path)

        written_again = generate(_psalms(), tmp_path)

        assert written_again == []

    def test_psalm_variant_broadcasts_the_same_vector(self, tmp_path):
        generate(_psalms(), tmp_path)

        table = pq.read_table(dataset_path(tmp_path, "phrase_det", "1gram_psalm"))
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert by_node[100] == by_node[101]
