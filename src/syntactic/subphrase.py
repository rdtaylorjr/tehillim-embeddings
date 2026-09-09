"""Masks subphrase `rela=par` to `NA`: the subphrase-level parallel-relation contaminant."""

from __future__ import annotations

QUARANTINED_SUBPHRASE_RELA = ("par",)

SAFE_SUBPHRASE_RELA_VOCABULARY: tuple[str, ...] = ("NA", "adj", "atr", "dem", "mod", "rec")


def mask_par(rela: str) -> str:
    """Returns `rela` unchanged, except `par` (and any other quarantined value) becomes `NA`."""
    return "NA" if rela in QUARANTINED_SUBPHRASE_RELA else rela


def half_verse_safe_subphrase_rela(half_verse_rela: tuple[str, ...]) -> tuple[str, ...]:
    """Applies `mask_par` to every subphrase in one half-verse."""
    return tuple(mask_par(value) for value in half_verse_rela)
