from __future__ import annotations

from lexical.corpus import LexicalPsalm
from lexical.export import dataset_path
from lexical.generate import generate


def _psalm(*, number, lexemes, forms, nodes):
    return LexicalPsalm(
        number=number,
        half_verse_lexemes=lexemes,
        half_verse_forms=forms,
        half_verse_nodes=nodes,
    )


def _psalms():
    return [
        _psalm(
            number=1, lexemes=(("A", "B"), ("A",)), forms=(("A0", "B0"), ("A0",)), nodes=(100, 101)
        ),
        _psalm(number=2, lexemes=(("C",),), forms=(("C0",),), nodes=(200,)),
    ]


class TestGenerate:
    def test_writes_both_vocabulary_variants(self, tmp_path):
        written = generate(_psalms(), tmp_path)

        assert set(written) == {"form_binary", "lexeme_binary"}
        assert dataset_path(tmp_path, "form", "binary").exists()
        assert dataset_path(tmp_path, "lexeme", "binary").exists()

    def test_skips_a_variant_whose_dataset_already_exists(self, tmp_path):
        generate(_psalms(), tmp_path)

        written_again = generate(_psalms(), tmp_path)

        assert written_again == []

    def test_lexeme_vocabulary_dimension_matches_distinct_lex_values(self, tmp_path):
        import pyarrow.parquet as pq

        generate(_psalms(), tmp_path)

        table = pq.read_table(dataset_path(tmp_path, "lexeme", "binary"))
        # distinct lex values across fixtures: A, B, C
        assert len(table["vector"].to_pylist()[0]) == 3

    def test_form_vocabulary_dimension_matches_distinct_lex0_values(self, tmp_path):
        import pyarrow.parquet as pq

        generate(_psalms(), tmp_path)

        table = pq.read_table(dataset_path(tmp_path, "form", "binary"))
        # distinct lex0 values across fixtures: A0, B0, C0
        assert len(table["vector"].to_pylist()[0]) == 3
