"""Builds the fixed lexical vocabulary (distinct lex or lex0 values) used to index colon vectors."""

from __future__ import annotations

from typing import Literal

from lexical.corpus import LexicalPsalm

VocabularyKey = Literal["lex", "lex0"]


def half_verses_for_key(psalm: LexicalPsalm, key: VocabularyKey) -> tuple[tuple[str, ...], ...]:
    """Selects a psalm's half-verse lex or lex0 sequences by `key`."""
    return psalm.half_verse_lexemes if key == "lex" else psalm.half_verse_forms


def build_vocabulary(psalms: list[LexicalPsalm], key: VocabularyKey) -> tuple[str, ...]:
    """Sorted distinct lex or lex0 values across every half-verse of every psalm."""
    values = {
        value
        for psalm in psalms
        for half_verse in half_verses_for_key(psalm, key)
        for value in half_verse
    }
    return tuple(sorted(values))
