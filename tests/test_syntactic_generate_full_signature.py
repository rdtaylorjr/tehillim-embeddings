from __future__ import annotations

from core.partition import BHSA_HALF_VERSE, Partition
from syntactic.corpus import PhrasePsalm
from syntactic.generate_full_signature import generate


def dataset_path(output_root, vocab, weight):
    partition = Partition(
        BHSA_HALF_VERSE, "syntactic", level="phrase", feature=vocab, construction=weight
    )
    return partition.file(output_root)


def _psalm(*, number, nodes, typ, function, det):
    return PhrasePsalm(
        number=number,
        half_verse_nodes=nodes,
        half_verse_typ=typ,
        half_verse_function=function,
        half_verse_det=det,
    )


def _psalms():
    return [
        _psalm(
            number=1,
            nodes=(100, 101),
            typ=(("NP", "VP"), ("PP",)),
            function=(("Subj", "Pred"), ("Cmpl",)),
            det=(("det", "NA"), ("und",)),
        )
    ]


def _external_counts():
    return {"NP:Subj:det": 5000, "VP:Pred": 5000, "PP:Cmpl:und": 5000}


class TestGenerate:
    def test_writes_every_declared_construction(self, tmp_path):
        written = generate(_psalms(), tmp_path, _external_counts(), k=1000)

        assert set(written) == {
            "phrase_full_signature_1gram",
            "phrase_full_signature_1gram_psalm",
            "phrase_full_signature_1_2gram",
            "phrase_full_signature_1_2gram_psalm",
            "phrase_full_signature_1_2_3gram",
            "phrase_full_signature_1_2_3gram_psalm",
        }
        assert dataset_path(tmp_path, "full_signature", "1gram").exists()
        assert dataset_path(tmp_path, "full_signature", "1_2_3gram").exists()

    def test_skips_variants_whose_dataset_already_exists(self, tmp_path):
        generate(_psalms(), tmp_path, _external_counts(), k=1000)

        written_again = generate(_psalms(), tmp_path, _external_counts(), k=1000)

        assert written_again == []
