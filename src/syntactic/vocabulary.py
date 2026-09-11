"""Frozen BHSA phrase vocabularies, verified against the raw `.tf` files, not their headers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from syntactic.corpus import PhrasePsalm

TYP_VOCABULARY: tuple[str, ...] = (
    "AdjP",
    "AdvP",
    "CP",
    "DPrP",
    "IPrP",
    "InjP",
    "InrP",
    "NP",
    "NegP",
    "PP",
    "PPrP",
    "PrNP",
    "VP",
)

FUNCTION_VOCABULARY: tuple[str, ...] = (
    "Adju",
    "Cmpl",
    "Conj",
    "EPPr",
    "ExsS",
    "Exst",
    "Frnt",
    "IntS",
    "Intj",
    "Loca",
    "ModS",
    "Modi",
    "NCoS",
    "NCop",
    "Nega",
    "Objc",
    "PrAd",
    "PrcS",
    "PreC",
    "PreO",
    "PreS",
    "Pred",
    "PtcO",
    "Ques",
    "Rela",
    "Subj",
    "Supp",
    "Time",
    "Voct",
)

DET_VOCABULARY: tuple[str, ...] = ("NA", "det", "und")


def typ_columns(psalm: PhrasePsalm) -> tuple[tuple[str, ...], ...]:
    """The phrase-atom type sequence per half-verse, which the typ n-grams read."""
    return psalm.half_verse_typ


def function_columns(psalm: PhrasePsalm) -> tuple[tuple[str, ...], ...]:
    """The phrase function sequence per half-verse, which the function n-grams read."""
    return psalm.half_verse_function


def det_columns(psalm: PhrasePsalm) -> tuple[tuple[str, ...], ...]:
    """The determination sequence per half-verse, which the det histogram reads."""
    return psalm.half_verse_det
