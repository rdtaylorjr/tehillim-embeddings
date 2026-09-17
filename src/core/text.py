"""Hebrew text-tier normalization shared by every corpus loader."""

from __future__ import annotations

from typing import Literal

#: The three text states a representation can be built over, ordered by how much pointing they keep.
TextTier = Literal["consonantal", "vocalized", "cantillation"]

#: Cantillation is U+0591-U+05AF; niqqud starts at U+05B0, so this range strips accents only.
_ACCENT_RANGE = range(0x0591, 0x05B0)


def strip_accents(text: str) -> str:
    """Removes cantillation marks from pointed text, keeping niqqud."""
    return "".join(ch for ch in text if ord(ch) not in _ACCENT_RANGE)


#: The Hebrew letters, alef to tav.
_LETTERS = range(0x05D0, 0x05EB)

#: Maqaf joins two words the corpus counts separately, so it becomes the space between them.
_MAQAF = "\u05be"


def consonantal(text: str) -> str:
    """Letters and spaces only: no points, no shin or sin dot, no maqaf, paseq or sof pasuq."""
    kept = (
        " " if ch == _MAQAF or ch.isspace() else ch if ord(ch) in _LETTERS else "" for ch in text
    )
    return " ".join("".join(kept).split())
