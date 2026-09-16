from __future__ import annotations

import pyarrow.parquet as pq

from core.partition import BHSA_HALF_VERSE, Partition
from syntactic.corpus import PhrasePsalm
from syntactic.generate_marginal import generate


def dataset_path(output_root, vocab, weight):
    partition = Partition(
        BHSA_HALF_VERSE, "syntactic", level="phrase", feature=vocab, construction=weight
    )
    return partition.file(output_root)


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


class TestGenerate:
    def test_writes_both_phrase_marginal_constructions(self, tmp_path):
        written = generate(_psalms(), tmp_path)

        assert set(written) == {
            "phrase_marginal_typ_function",
            "phrase_marginal_typ_function_psalm",
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
