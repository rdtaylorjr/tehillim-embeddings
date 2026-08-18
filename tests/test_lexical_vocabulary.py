from __future__ import annotations

from lexical.corpus import LexicalPsalm
from lexical.vocabulary import build_vocabulary


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
