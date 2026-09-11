"""Builds the fixed lexical vocabulary (distinct lex or lex0 values) indexing the vectors."""

from __future__ import annotations

from typing import Literal

from core.columns import PsalmColumns
from core.vocabulary import sorted_distinct
from lexical.corpus import LexicalPsalm

VocabularyKey = Literal["lex", "lex0"]


def half_verses_for_key(psalm: LexicalPsalm, key: VocabularyKey) -> tuple[tuple[str, ...], ...]:
    """Selects a psalm's half-verse lex or lex0 sequences by `key`."""
    return psalm.half_verse_lexemes if key == "lex" else psalm.half_verse_forms


def columns_for_key(psalms: list[LexicalPsalm], key: VocabularyKey) -> list[PsalmColumns]:
    """Projects psalms onto the shared column view, selecting lex or lex0 sequences."""
    return [
        PsalmColumns(psalm.number, psalm.half_verse_nodes, half_verses_for_key(psalm, key))
        for psalm in psalms
    ]


def build_vocabulary(psalms: list[LexicalPsalm], key: VocabularyKey) -> tuple[str, ...]:
    """Sorted distinct lex or lex0 values across every half-verse of every psalm."""
    return sorted_distinct(
        half_verse for psalm in psalms for half_verse in half_verses_for_key(psalm, key)
    )
