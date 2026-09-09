"""Loads BHSA surface word-form features per half-verse, in three text tiers, via Text-Fabric."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.corpus import DEFAULT_BHSA_CLONE, BaseCorpus, load_api
from core.text import strip_accents

__all__ = ["DEFAULT_BHSA_CLONE", "SurfaceCorpus", "SurfacePsalm"]

_REQUIRED_FEATURES = "otype book chapter verse g_cons_utf8 g_word_utf8"


@dataclass(frozen=True, slots=True)
class SurfacePsalm:
    """One psalm's half-verse surface word-form sequences, in three text tiers, aligned words."""

    number: int
    half_verse_nodes: tuple[int, ...] = ()
    half_verse_consonantal: tuple[tuple[str, ...], ...] = ()
    half_verse_vocalized: tuple[tuple[str, ...], ...] = ()
    half_verse_cantillation: tuple[tuple[str, ...], ...] = ()


class SurfaceCorpus(BaseCorpus[SurfacePsalm]):
    """A loaded BHSA Text-Fabric corpus, scoped to half-verse surface-form extraction."""

    @classmethod
    def load(
        cls, tf_path: Path | None = None, *, loader: Callable[..., Any] = load_api
    ) -> SurfaceCorpus:
        """Loads BHSA from `tf_path`, else $TEHILLIM_BHSA_PATH, else `DEFAULT_BHSA_CLONE`."""
        return cls(loader(tf_path, _REQUIRED_FEATURES))

    def _extract(self, number: int, half_verse_nodes: tuple[int, ...]) -> SurfacePsalm:
        """Reads each half-verse's consonantal, niqqud-only, and fully pointed word forms."""
        F, L = self._api.F, self._api.L  # noqa: N806
        words_by_half_verse = [L.d(hv, otype="word") for hv in half_verse_nodes]
        return SurfacePsalm(
            number=number,
            half_verse_consonantal=tuple(
                tuple(F.g_cons_utf8.v(w) for w in words) for words in words_by_half_verse
            ),
            half_verse_vocalized=tuple(
                tuple(strip_accents(F.g_word_utf8.v(w)) for w in words)
                for words in words_by_half_verse
            ),
            half_verse_cantillation=tuple(
                tuple(F.g_word_utf8.v(w) for w in words) for words in words_by_half_verse
            ),
            half_verse_nodes=half_verse_nodes,
        )
