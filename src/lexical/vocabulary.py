"""Builds the fixed lexical vocabulary (distinct lex or lex0 values) used to index colon vectors."""

from __future__ import annotations

from typing import Literal

from lexical.corpus import LexicalPsalm

VocabularyKey = Literal["lex", "lex0"]


def cola_for_key(psalm: LexicalPsalm, key: VocabularyKey) -> tuple[tuple[str, ...], ...]:
    """Selects a psalm's colon lex or lex0 sequences by `key`."""
    return psalm.colon_lexemes if key == "lex" else psalm.colon_forms


def build_vocabulary(psalms: list[LexicalPsalm], key: VocabularyKey) -> tuple[str, ...]:
    """Sorted distinct lex or lex0 values across every colon of every psalm."""
    values = {value for psalm in psalms for colon in cola_for_key(psalm, key) for value in colon}
    return tuple(sorted(values))
