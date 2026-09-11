from __future__ import annotations

from core.export import dataset_path as _dataset_path
from syntactic.corpus import PhrasePsalm
from syntactic.generate_full_signature import generate


def dataset_path(output_root, vocab, weight):
    return _dataset_path(
        output_root, vocab, weight, domain="syntactic", unit_key="feature", level="phrase"
    )


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
            "full_signature_1gram",
            "full_signature_1gram_psalm",
            "full_signature_1_2gram",
            "full_signature_1_2gram_psalm",
            "full_signature_1_2_3gram",
            "full_signature_1_2_3gram_psalm",
        }
        assert dataset_path(tmp_path, "full_signature", "1gram").exists()
        assert dataset_path(tmp_path, "full_signature", "1_2_3gram").exists()

    def test_skips_variants_whose_dataset_already_exists(self, tmp_path):
        generate(_psalms(), tmp_path, _external_counts(), k=1000)

        written_again = generate(_psalms(), tmp_path, _external_counts(), k=1000)

        assert written_again == []
