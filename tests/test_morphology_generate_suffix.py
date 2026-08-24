from __future__ import annotations

from lexical.export import dataset_path as _dataset_path
from morphology.corpus import MorphologicalPsalm
from morphology.generate_suffix import generate


def dataset_path(output_root, vocab, weight):
    return _dataset_path(output_root, vocab, weight, domain="morphology", unit_key="feature")


def _psalm(*, number, nodes, **feature_columns):
    return MorphologicalPsalm(
        number=number,
        colon_nodes=nodes,
        **{f"colon_{feature}": values for feature, values in feature_columns.items()},
    )


def _psalms():
    return [
        _psalm(
            number=1,
            nodes=(100, 101),
            sp=(("subs", "verb"), ("subs",)),
            gn=(("m", "NA"), ("f",)),
            nu=(("sg", "NA"), ("sg",)),
            ps=(("NA", "p3"), ("NA",)),
            st=(("a", "NA"), ("a",)),
            vs=(("NA", "qal"), ("NA",)),
            vt=(("NA", "perf"), ("NA",)),
            prs_gn=(("NA", "NA"), ("m",)),
            prs_nu=(("NA", "NA"), ("pl",)),
            prs_ps=(("NA", "NA"), ("p3",)),
        ),
    ]


def _external_counts():
    return {"subs|m|sg|a": 5000, "verb|qal|perf|p3": 5000, "subs|f|sg|a": 5000}


class TestGenerate:
    def test_writes_all_four_morph_suffix_variants(self, tmp_path):
        written = generate(_psalms(), tmp_path, _external_counts(), k=1000)

        for construction in (
            "inventory",
            "inventory_psalm",
            "host_plus_suffix",
            "host_plus_suffix_psalm",
        ):
            assert f"morph_suffix_{construction}" in written
            assert dataset_path(tmp_path, "morph_suffix", construction).exists()

    def test_skips_variants_whose_dataset_already_exists(self, tmp_path):
        generate(_psalms(), tmp_path, _external_counts(), k=1000)

        written_again = generate(_psalms(), tmp_path, _external_counts(), k=1000)

        assert written_again == []
