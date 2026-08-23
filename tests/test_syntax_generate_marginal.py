from __future__ import annotations

import pyarrow.parquet as pq

from lexical.export import dataset_path as _dataset_path
from syntax.corpus import PhrasePsalm
from syntax.generate_marginal import generate


def dataset_path(output_root, vocab, weight):
    return _dataset_path(
        output_root, vocab, weight, domain="syntax", unit_key="feature", level="phrase"
    )


def _psalm(*, number, nodes, typ, function):
    return PhrasePsalm(number=number, colon_nodes=nodes, colon_typ=typ, colon_function=function)


def _psalms():
    return [
        _psalm(
            number=1,
            nodes=(100, 101),
            typ=(("NP", "VP"), ("PP",)),
            function=(("Subj", "Pred"), ("Cmpl",)),
        ),
    ]


class TestGenerate:
    def test_writes_both_phrase_marginal_constructions(self, tmp_path):
        written = generate(_psalms(), tmp_path)

        assert set(written) == {
            "marginal_typ_function",
            "marginal_typ_function_psalm",
        }
        assert dataset_path(tmp_path, "marginal", "typ_function").exists()
        assert dataset_path(tmp_path, "marginal", "typ_function_psalm").exists()

    def test_skips_variants_whose_dataset_already_exists(self, tmp_path):
        generate(_psalms(), tmp_path)

        written_again = generate(_psalms(), tmp_path)

        assert written_again == []

    def test_psalm_variant_broadcasts_the_same_vector_within_a_psalm(self, tmp_path):
        generate(_psalms(), tmp_path)

        table = pq.read_table(dataset_path(tmp_path, "marginal", "typ_function_psalm"))
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert by_node[100] == by_node[101]
