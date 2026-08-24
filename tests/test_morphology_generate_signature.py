from __future__ import annotations

import pyarrow.parquet as pq

from lexical.export import dataset_path as _dataset_path
from morphology.corpus import MorphologicalPsalm
from morphology.generate_signature import generate


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


def _external_counts():
    return {"subs|m|sg|a": 5000, "verb|qal|perf|p3": 5000, "subs|f|sg|a": 5000}


class TestGenerate:
    def test_writes_morph_atomic_colon_and_psalm(self, tmp_path):
        written = generate(_psalms(), tmp_path, _external_counts(), k=1000)

        assert "morph_atomic_core" in written
        assert "morph_atomic_core_psalm" in written
        assert dataset_path(tmp_path, "morph_atomic", "core").exists()
        assert dataset_path(tmp_path, "morph_atomic", "core_psalm").exists()

    def test_writes_all_six_morph_signature_variants(self, tmp_path):
        written = generate(_psalms(), tmp_path, _external_counts(), k=1000)

        for construction in (
            "inventory",
            "inventory_psalm",
            "1_2gram",
            "1_2gram_psalm",
            "1_2_3gram",
            "1_2_3gram_psalm",
        ):
            assert f"morph_signature_{construction}" in written
            assert dataset_path(tmp_path, "morph_signature", construction).exists()

    def test_skips_variants_whose_dataset_already_exists(self, tmp_path):
        generate(_psalms(), tmp_path, _external_counts(), k=1000)

        written_again = generate(_psalms(), tmp_path, _external_counts(), k=1000)

        assert written_again == []

    def test_1_2_3gram_is_written_with_the_sparse_schema_not_dense(self, tmp_path):
        generate(_psalms(), tmp_path, _external_counts(), k=1000)

        table = pq.read_table(dataset_path(tmp_path, "morph_signature", "1_2_3gram"))
        assert set(table.column_names) == {"node_id", "indices", "values"}
        assert table.schema.metadata[b"sparse"] == b"true"

    def test_1_2gram_is_still_written_with_the_dense_schema(self, tmp_path):
        generate(_psalms(), tmp_path, _external_counts(), k=1000)

        table = pq.read_table(dataset_path(tmp_path, "morph_signature", "1_2gram"))
        assert set(table.column_names) == {"node_id", "vector"}
