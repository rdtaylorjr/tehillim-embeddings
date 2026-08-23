"""Frozen BHSA phrase vocabularies, verified against the raw `.tf` files, not their headers."""

from __future__ import annotations

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
