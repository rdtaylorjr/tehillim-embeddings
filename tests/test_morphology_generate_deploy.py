from __future__ import annotations

from lexical.export import dataset_path as _dataset_path
from morphology.corpus import MorphologicalPsalm
from morphology.generate_deploy import generate


def dataset_path(output_root, vocab, weight):
    return _dataset_path(output_root, vocab, weight, domain="morphology", unit_key="feature")


def _psalm(*, number, nodes, **feature_columns):
    return MorphologicalPsalm(
        number=number,
        half_verse_nodes=nodes,
        **{f"half_verse_{feature}": values for feature, values in feature_columns.items()},
    )


def _psalms():
    return [
        _psalm(
            number=1,
            nodes=(100, 101),
            prs_gn=(("NA",), ("m",)),
            prs_nu=(("NA",), ("pl",)),
            prs_ps=(("NA",), ("p3",)),
        ),
    ]


class TestGenerate:
    def test_writes_the_morph_suffix_posmean_dataset(self, tmp_path):
        written = generate(_psalms(), tmp_path)

        assert written == ["morph_suffix_posmean"]
        assert dataset_path(tmp_path, "morph_suffix", "posmean").exists()

    def test_skips_when_the_dataset_already_exists(self, tmp_path):
        generate(_psalms(), tmp_path)

        written_again = generate(_psalms(), tmp_path)

        assert written_again == []
