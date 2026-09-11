from __future__ import annotations

import pyarrow.parquet as pq

from core.export import dataset_path as _dataset_path
from syntactic.corpus import PhrasePsalm
from syntactic.generate_rela import generate
from syntactic.rela import SAFE_RELA_VOCABULARY


def dataset_path(output_root, vocab, weight):
    return _dataset_path(
        output_root, vocab, weight, domain="syntactic", unit_key="feature", level="phrase"
    )


def _psalm(*, number, nodes, rela):
    return PhrasePsalm(number=number, half_verse_nodes=nodes, half_verse_rela=rela)


def _psalms():
    return [_psalm(number=1, nodes=(100, 101), rela=(("NA", "Para"), ("Appo",)))]


class TestGenerate:
    def test_writes_every_declared_construction(self, tmp_path):
        written = generate(_psalms(), tmp_path)

        assert set(written) == {
            "rela_1gram",
            "rela_1gram_psalm",
            "rela_1_2gram",
            "rela_1_2gram_psalm",
            "rela_1_2_3gram",
            "rela_1_2_3gram_psalm",
        }
        assert dataset_path(tmp_path, "rela", "1gram").exists()
        assert dataset_path(tmp_path, "rela", "1gram_psalm").exists()
        assert dataset_path(tmp_path, "rela", "1_2gram").exists()
        assert dataset_path(tmp_path, "rela", "1_2gram_psalm").exists()
        assert dataset_path(tmp_path, "rela", "1_2_3gram").exists()
        assert dataset_path(tmp_path, "rela", "1_2_3gram_psalm").exists()

    def test_skips_variants_whose_dataset_already_exists(self, tmp_path):
        generate(_psalms(), tmp_path)

        written_again = generate(_psalms(), tmp_path)

        assert written_again == []

    def test_written_vectors_never_show_mass_outside_the_safe_vocabulary_dimension(self, tmp_path):
        generate(_psalms(), tmp_path)

        table = pq.read_table(dataset_path(tmp_path, "rela", "1gram"))
        for vector in table["vector"].to_pylist():
            assert len(vector) == len(SAFE_RELA_VOCABULARY)
