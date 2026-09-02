"""Builds the fixed surface-form vocabulary (distinct word forms) indexing the vectors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from core.columns import PsalmColumns
from lexical.surface_corpus import SurfacePsalm

SurfaceTier = Literal["consonantal", "vocalized", "cantillation"]


#: Written out rather than resolved by name so a renamed field fails type checking.
_HALF_VERSES_BY_TIER: dict[SurfaceTier, Callable[[SurfacePsalm], tuple[tuple[str, ...], ...]]] = {
    "consonantal": lambda psalm: psalm.half_verse_consonantal,
    "vocalized": lambda psalm: psalm.half_verse_vocalized,
    "cantillation": lambda psalm: psalm.half_verse_cantillation,
}


def half_verses_for_tier(psalm: SurfacePsalm, tier: SurfaceTier) -> tuple[tuple[str, ...], ...]:
    """Selects a psalm's half-verse surface-form sequences for one text tier."""
    return _HALF_VERSES_BY_TIER[tier](psalm)


def columns_for_tier(psalms: list[SurfacePsalm], tier: SurfaceTier) -> list[PsalmColumns]:
    """Projects psalms onto the shared column view, selecting one text tier's word forms."""
    return [
        PsalmColumns(psalm.number, psalm.half_verse_nodes, half_verses_for_tier(psalm, tier))
        for psalm in psalms
    ]


def build_surface_vocabulary(psalms: list[SurfacePsalm], tier: SurfaceTier) -> tuple[str, ...]:
    """Sorted distinct surface word forms across every half-verse of every psalm, at one tier."""
    values = {
        value
        for psalm in psalms
        for half_verse in half_verses_for_tier(psalm, tier)
        for value in half_verse
    }
    return tuple(sorted(values))
