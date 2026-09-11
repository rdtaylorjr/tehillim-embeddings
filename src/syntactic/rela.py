"""Masks `rela=Para` to `NA` before histogramming: quarantines the parallel-relation contaminant."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from syntactic.corpus import PhrasePsalm

QUARANTINED_RELA = ("Para",)

SAFE_RELA_VOCABULARY: tuple[str, ...] = ("NA", "Appo", "Link", "Sfxs", "Spec")


def mask_para(rela: str) -> str:
    """Returns `rela` unchanged, except `Para` (and any other quarantined value) becomes `NA`."""
    return "NA" if rela in QUARANTINED_RELA else rela


def half_verse_safe_rela(half_verse_rela: tuple[str, ...]) -> tuple[str, ...]:
    """Applies `mask_para` to every phrase atom in one half-verse."""
    return tuple(mask_para(value) for value in half_verse_rela)


def safe_rela_columns(psalm: PhrasePsalm) -> tuple[tuple[str, ...], ...]:
    """The phrase-relation sequence per half-verse, `Para` masked before any counting."""
    return tuple(half_verse_safe_rela(column) for column in psalm.half_verse_rela)
