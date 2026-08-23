from __future__ import annotations

import pyarrow.parquet as pq

from lexical.export import dataset_path as _dataset_path
from syntax.corpus import PhrasePsalm
from syntax.generate_subphrase import generate
from syntax.subphrase import SAFE_SUBPHRASE_RELA_VOCABULARY


def dataset_path(output_root, vocab, weight):
    return _dataset_path(
        output_root, vocab, weight, domain="syntax", unit_key="feature", level="phrase"
    )


def _psalm(*, number, nodes, rela):
    return PhrasePsalm(number=number, half_verse_nodes=nodes, half_verse_subphrase_rela=rela)


def _psalms():
    return [_psalm(number=1, nodes=(100, 101), rela=(("NA", "par"), ("rec",)))]


class TestGenerate:
    def test_writes_both_constructions(self, tmp_path):
        written = generate(_psalms(), tmp_path)

        assert set(written) == {"subphrase_rela_1gram", "subphrase_rela_1gram_psalm"}
        assert dataset_path(tmp_path, "subphrase_rela", "1gram").exists()
        assert dataset_path(tmp_path, "subphrase_rela", "1gram_psalm").exists()

    def test_skips_variants_whose_dataset_already_exists(self, tmp_path):
        generate(_psalms(), tmp_path)

        written_again = generate(_psalms(), tmp_path)

        assert written_again == []

    def test_written_vectors_never_show_mass_outside_the_safe_vocabulary_dimension(self, tmp_path):
        generate(_psalms(), tmp_path)

        table = pq.read_table(dataset_path(tmp_path, "subphrase_rela", "1gram"))
        for vector in table["vector"].to_pylist():
            assert len(vector) == len(SAFE_SUBPHRASE_RELA_VOCABULARY)
