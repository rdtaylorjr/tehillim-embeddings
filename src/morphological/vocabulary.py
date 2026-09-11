"""Frozen BHSA morphology vocabularies, verified against the raw `.tf` files, not their headers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from morphological.corpus import MorphologicalPsalm

SP_VOCABULARY: tuple[str, ...] = (
    "adjv",
    "advb",
    "art",
    "conj",
    "inrg",
    "intj",
    "nega",
    "nmpr",
    "prde",
    "prep",
    "prin",
    "prps",
    "subs",
    "verb",
)

GN_VOCABULARY: tuple[str, ...] = ("NA", "f", "m", "unknown")

NU_VOCABULARY: tuple[str, ...] = ("NA", "du", "pl", "sg", "unknown")

PS_VOCABULARY: tuple[str, ...] = ("NA", "p1", "p2", "p3", "unknown")

ST_VOCABULARY: tuple[str, ...] = ("NA", "a", "c", "e")

VS_VOCABULARY: tuple[str, ...] = (
    "NA",
    "afel",
    "etpa",
    "etpe",
    "haf",
    "hif",
    "hit",
    "hof",
    "hotp",
    "hsht",
    "htpa",
    "htpe",
    "htpo",
    "nif",
    "nit",
    "pasq",
    "peal",
    "peil",
    "piel",
    "poal",
    "poel",
    "pual",
    "qal",
    "shaf",
    "tif",
)

VT_VOCABULARY: tuple[str, ...] = (
    "NA",
    "impf",
    "impv",
    "infa",
    "infc",
    "perf",
    "ptca",
    "ptcp",
    "wayq",
)

PRS_GN_VOCABULARY: tuple[str, ...] = ("NA", "f", "m", "unknown")

PRS_NU_VOCABULARY: tuple[str, ...] = ("NA", "pl", "sg")

PRS_PS_VOCABULARY: tuple[str, ...] = ("NA", "p1", "p2", "p3")


def sp_columns(psalm: MorphologicalPsalm) -> tuple[tuple[str, ...], ...]:
    """The part-of-speech sequence per half-verse, which the sp n-grams read."""
    return psalm.half_verse_sp
