"""Hebrew text-tier normalization shared by every corpus loader."""

from __future__ import annotations

#: Cantillation is U+0591-U+05AF; niqqud starts at U+05B0, so this range strips accents only.
_ACCENT_RANGE = range(0x0591, 0x05B0)


def strip_accents(text: str) -> str:
    """Removes cantillation marks from pointed text, keeping niqqud."""
    return "".join(ch for ch in text if ord(ch) not in _ACCENT_RANGE)
