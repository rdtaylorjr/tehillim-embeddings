"""Builds the fixed surface-form vocabulary (distinct word forms) used to index colon vectors."""

from __future__ import annotations

from typing import Literal

from lexical.surface_corpus import SurfacePsalm

SurfaceTier = Literal["consonantal", "vocalized", "cantillation"]

_FIELD_BY_TIER = {
    "consonantal": "half_verse_consonantal",
    "vocalized": "half_verse_vocalized",
    "cantillation": "half_verse_cantillation",
}


def half_verses_for_tier(psalm: SurfacePsalm, tier: SurfaceTier) -> tuple[tuple[str, ...], ...]:
    """Selects a psalm's half-verse surface-form sequences by text `tier`."""
    return getattr(psalm, _FIELD_BY_TIER[tier])  # type: ignore[no-any-return]


def build_surface_vocabulary(psalms: list[SurfacePsalm], tier: SurfaceTier) -> tuple[str, ...]:
    """Sorted distinct surface word forms across every half-verse of every psalm, at one tier."""
    values = {
        value
        for psalm in psalms
        for half_verse in half_verses_for_tier(psalm, tier)
        for value in half_verse
    }
    return tuple(sorted(values))
