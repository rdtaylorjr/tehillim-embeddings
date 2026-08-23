from __future__ import annotations

from lexical.export import dataset_path as _dataset_path
from morphology.corpus import MorphologicalPsalm
from morphology.generate_morphology import _FEATURES, generate


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
            prs_gn=(("NA", "NA"), ("NA",)),
            prs_nu=(("NA", "NA"), ("NA",)),
            prs_ps=(("NA", "NA"), ("NA",)),
        ),
    ]


class TestGenerate:
    def test_writes_atomic_and_sp_plus_variants_for_all_nine_features(self, tmp_path):
        written = generate(_psalms(), tmp_path)

        expected = set()
        for feature in _FEATURES:
            expected.update(
                {
                    f"morph_{feature}_atomic",
                    f"morph_{feature}_atomic_psalm",
                    f"morph_{feature}_sp_plus",
                    f"morph_{feature}_sp_plus_psalm",
                }
            )
        expected.update({"morph_full_all", "morph_full_all_psalm"})
        assert set(written) == expected

    def test_all_datasets_exist_on_disk(self, tmp_path):
        generate(_psalms(), tmp_path)

        for feature in _FEATURES:
            assert dataset_path(tmp_path, f"morph_{feature}", "atomic").exists()
            assert dataset_path(tmp_path, f"morph_{feature}", "atomic_psalm").exists()
            assert dataset_path(tmp_path, f"morph_{feature}", "sp_plus").exists()
            assert dataset_path(tmp_path, f"morph_{feature}", "sp_plus_psalm").exists()
        assert dataset_path(tmp_path, "morph_full", "all").exists()
        assert dataset_path(tmp_path, "morph_full", "all_psalm").exists()

    def test_skips_variants_whose_dataset_already_exists(self, tmp_path):
        generate(_psalms(), tmp_path)

        written_again = generate(_psalms(), tmp_path)

        assert written_again == []
