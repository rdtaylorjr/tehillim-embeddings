"""Whole-Hebrew-Bible token frequency per surface word form, one text tier at a time."""

from __future__ import annotations

from typing import Any

from lexical.surface_corpus import _strip_accents
from lexical.surface_vocabulary import SurfaceTier


def surface_token_frequencies(api: Any, tier: SurfaceTier) -> dict[str, int]:
    """Whole-Bible token count per surface form at one text tier, not just its Psalms subset."""
    F = api.F  # noqa: N806
    frequencies: dict[str, int] = {}
    for word in F.otype.s("word"):
        if tier == "consonantal":
            value = F.g_cons_utf8.v(word)
        elif tier == "cantillation":
            value = F.g_word_utf8.v(word)
        else:
            value = _strip_accents(F.g_word_utf8.v(word))
        frequencies[value] = frequencies.get(value, 0) + 1
    return frequencies
