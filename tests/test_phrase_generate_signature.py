from __future__ import annotations

from lexical.export import dataset_path as _dataset_path
from phrase.corpus import PhrasePsalm
from phrase.generate_signature import generate


def dataset_path(output_root, vocab, weight):
    return _dataset_path(output_root, vocab, weight, dataset_type="phrase")


def _psalm(*, number, nodes, typ, function):
    return PhrasePsalm(
        number=number, half_verse_nodes=nodes, half_verse_typ=typ, half_verse_function=function
    )


def _psalms():
    return [
        _psalm(
            number=1,
            nodes=(100, 101),
            typ=(("NP", "VP"), ("PP",)),
            function=(("Subj", "Pred"), ("Cmpl",)),
        ),
    ]


def _external_counts():
    return {"NP:Subj": 5000, "VP:Pred": 5000, "PP:Cmpl": 5000}


class TestGenerate:
    def test_writes_all_six_phrase_signature_variants(self, tmp_path):
        written = generate(_psalms(), tmp_path, _external_counts(), k=1000)

        for construction in (
            "inventory",
            "inventory_psalm",
            "1_2gram",
            "1_2gram_psalm",
            "1_2_3gram",
            "1_2_3gram_psalm",
        ):
            assert f"phrase_signature_{construction}" in written
            assert dataset_path(tmp_path, "phrase_signature", construction).exists()

    def test_skips_variants_whose_dataset_already_exists(self, tmp_path):
        generate(_psalms(), tmp_path, _external_counts(), k=1000)

        written_again = generate(_psalms(), tmp_path, _external_counts(), k=1000)

        assert written_again == []
