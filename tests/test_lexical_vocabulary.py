from __future__ import annotations

from lexical.corpus import LexicalPsalm
from lexical.vocabulary import build_vocabulary, columns_for_key


def _psalm(*, number, lexemes, forms):
    return LexicalPsalm(
        number=number,
        half_verse_lexemes=lexemes,
        half_verse_forms=forms,
        half_verse_nodes=tuple(range(len(lexemes))),
    )


class TestBuildVocabulary:
    def test_returns_sorted_distinct_lex_values_across_all_psalms(self):
        psalms = [
            _psalm(number=1, lexemes=(("B", "W"),), forms=(("B", "W"),)),
            _psalm(number=2, lexemes=(("A", "B"),), forms=(("A", "B"),)),
        ]

        vocabulary = build_vocabulary(psalms, key="lex")

        assert vocabulary == ("A", "B", "W")

    def test_returns_sorted_distinct_lex0_values_across_all_psalms(self):
        psalms = [
            _psalm(number=1, lexemes=(("B", "W"),), forms=(("B0", "W0"),)),
            _psalm(number=2, lexemes=(("A", "B"),), forms=(("A0", "B0"),)),
        ]

        vocabulary = build_vocabulary(psalms, key="lex0")

        assert vocabulary == ("A0", "B0", "W0")

    def test_lex0_collapses_homonyms_that_lex_keeps_distinct(self):
        # Two distinct lex values (BR>[ and BR>=[) share one lex0 (BR>).
        psalms = [
            _psalm(number=1, lexemes=(("BR>[", "BR>=["),), forms=(("BR>", "BR>"),)),
        ]

        lex_vocab = build_vocabulary(psalms, key="lex")
        lex0_vocab = build_vocabulary(psalms, key="lex0")

        assert lex_vocab == ("BR>=[", "BR>[")
        assert lex0_vocab == ("BR>",)
        assert len(lex0_vocab) < len(lex_vocab)

    def test_repeated_lexemes_within_and_across_half_verses_count_once(self):
        psalms = [
            _psalm(number=1, lexemes=(("A", "A"), ("A",)), forms=(("A0", "A0"), ("A0",))),
        ]

        assert build_vocabulary(psalms, key="lex") == ("A",)

    def test_empty_psalm_list_returns_empty_vocabulary(self):
        assert build_vocabulary([], key="lex") == ()


class TestColumnsForKey:
    def test_projects_each_psalm_onto_number_nodes_and_that_key_s_half_verses(self):
        psalms = [
            LexicalPsalm(
                number=7,
                half_verse_lexemes=(("A", "B"),),
                half_verse_forms=(("A0", "B0"),),
                half_verse_nodes=(100,),
            )
        ]

        columns = columns_for_key(psalms, "lex0")

        assert len(columns) == 1
        assert columns[0].number == 7
        assert columns[0].nodes == (100,)
        assert columns[0].half_verses == (("A0", "B0"),)

    def test_lex_and_lex0_select_different_sequences(self):
        psalms = [
            LexicalPsalm(
                number=1,
                half_verse_lexemes=(("A",),),
                half_verse_forms=(("A0",),),
                half_verse_nodes=(100,),
            )
        ]

        assert columns_for_key(psalms, "lex")[0].half_verses == (("A",),)
        assert columns_for_key(psalms, "lex0")[0].half_verses == (("A0",),)

    def test_preserves_psalm_order(self):
        psalms = [
            LexicalPsalm(
                number=n,
                half_verse_lexemes=(("A",),),
                half_verse_forms=(("A0",),),
                half_verse_nodes=(n,),
            )
            for n in (3, 1, 2)
        ]

        assert [c.number for c in columns_for_key(psalms, "lex")] == [3, 1, 2]
