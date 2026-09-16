from __future__ import annotations

import pyarrow.parquet as pq

from core.partition import BHSA_HALF_VERSE, Partition
from syntactic.corpus import PhrasePsalm
from syntactic.generate_subphrase import generate
from syntactic.subphrase import SAFE_SUBPHRASE_RELA_VOCABULARY


def dataset_path(output_root, vocab, weight):
    partition = Partition(
        BHSA_HALF_VERSE, "syntactic", level="phrase", feature=vocab, construction=weight
    )
    return partition.file(output_root)


def _psalm(*, number, nodes, rela):
    return PhrasePsalm(number=number, half_verse_nodes=nodes, half_verse_subphrase_rela=rela)


def _psalms():
    return [_psalm(number=1, nodes=(100, 101), rela=(("NA", "par"), ("rec",)))]


class TestGenerate:
    def test_writes_every_declared_construction(self, tmp_path):
        written = generate(_psalms(), tmp_path)

        assert set(written) == {
            "phrase_subphrase_rela_1gram",
            "phrase_subphrase_rela_1gram_psalm",
            "phrase_subphrase_rela_1_2gram",
            "phrase_subphrase_rela_1_2gram_psalm",
            "phrase_subphrase_rela_1_2_3gram",
            "phrase_subphrase_rela_1_2_3gram_psalm",
        }
        assert dataset_path(tmp_path, "subphrase_rela", "1gram").exists()
        assert dataset_path(tmp_path, "subphrase_rela", "1gram_psalm").exists()
        assert dataset_path(tmp_path, "subphrase_rela", "1_2gram").exists()
        assert dataset_path(tmp_path, "subphrase_rela", "1_2gram_psalm").exists()
        assert dataset_path(tmp_path, "subphrase_rela", "1_2_3gram").exists()
        assert dataset_path(tmp_path, "subphrase_rela", "1_2_3gram_psalm").exists()

    def test_skips_variants_whose_dataset_already_exists(self, tmp_path):
        generate(_psalms(), tmp_path)

        written_again = generate(_psalms(), tmp_path)

        assert written_again == []

    def test_written_vectors_never_show_mass_outside_the_safe_vocabulary_dimension(self, tmp_path):
        generate(_psalms(), tmp_path)

        table = pq.read_table(dataset_path(tmp_path, "subphrase_rela", "1gram"))
        for vector in table["vector"].to_pylist():
            assert len(vector) == len(SAFE_SUBPHRASE_RELA_VOCABULARY)
