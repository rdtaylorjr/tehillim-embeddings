from __future__ import annotations

from lexical.export import dataset_path as _dataset_path
from syntax.corpus import PhrasePsalm
from syntax.generate_deploy import generate


def dataset_path(output_root, vocab, weight):
    return _dataset_path(
        output_root, vocab, weight, domain="syntax", unit_key="feature", level="phrase"
    )


def _psalm(*, number, nodes, typ, function):
    return PhrasePsalm(
        number=number, half_verse_nodes=nodes, half_verse_typ=typ, half_verse_function=function
    )


def _psalms():
    return [
        _psalm(
            number=1,
            nodes=(100, 101),
            typ=(("NP",), ("VP",)),
            function=(("Subj",), ("Pred",)),
        ),
    ]


def _external_counts():
    return {"NP:Subj": 5000, "VP:Pred": 5000}


class TestGenerate:
    def test_writes_the_phrase_signature_posmean_dataset(self, tmp_path):
        written = generate(_psalms(), tmp_path, _external_counts(), k=1000)

        assert written == ["signature_posmean"]
        assert dataset_path(tmp_path, "signature", "posmean").exists()

    def test_skips_when_the_dataset_already_exists(self, tmp_path):
        generate(_psalms(), tmp_path, _external_counts(), k=1000)

        written_again = generate(_psalms(), tmp_path, _external_counts(), k=1000)

        assert written_again == []
