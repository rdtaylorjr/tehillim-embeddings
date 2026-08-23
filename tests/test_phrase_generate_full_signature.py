from __future__ import annotations

from lexical.export import dataset_path as _dataset_path
from phrase.corpus import PhrasePsalm
from phrase.generate_full_signature import generate


def dataset_path(output_root, vocab, weight):
    return _dataset_path(output_root, vocab, weight, dataset_type="phrase")


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
    def test_writes_both_constructions(self, tmp_path):
        written = generate(_psalms(), tmp_path, _external_counts(), k=1000)

        assert set(written) == {
            "phrase_full_signature_inventory",
            "phrase_full_signature_inventory_psalm",
        }
        assert dataset_path(tmp_path, "phrase_full_signature", "inventory").exists()
        assert dataset_path(tmp_path, "phrase_full_signature", "inventory_psalm").exists()

    def test_skips_variants_whose_dataset_already_exists(self, tmp_path):
        generate(_psalms(), tmp_path, _external_counts(), k=1000)

        written_again = generate(_psalms(), tmp_path, _external_counts(), k=1000)

        assert written_again == []
