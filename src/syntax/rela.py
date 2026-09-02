"""Masks `rela=Para` to `NA` before histogramming: quarantines the parallel-relation contaminant."""

from __future__ import annotations

QUARANTINED_RELA = ("Para",)

SAFE_RELA_VOCABULARY: tuple[str, ...] = ("NA", "Appo", "Link", "Sfxs", "Spec")


def mask_para(rela: str) -> str:
    """Returns `rela` unchanged, except `Para` (and any other quarantined value) becomes `NA`."""
    return "NA" if rela in QUARANTINED_RELA else rela


def half_verse_safe_rela(half_verse_rela: tuple[str, ...]) -> tuple[str, ...]:
    """Applies `mask_para` to every phrase atom in one half-verse."""
    return tuple(mask_para(value) for value in half_verse_rela)
