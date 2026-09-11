"""Loads Hebrew Psalms half-verse text and BHSA node ids via Text-Fabric."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.corpus import DEFAULT_BHSA_CLONE, BaseCorpus, shared_api
from core.text import strip_accents

__all__ = ["DEFAULT_BHSA_CLONE", "Corpus", "SemanticPsalm"]

_REQUIRED_FEATURES = "otype book chapter verse g_word_utf8 g_cons_utf8 trailer_utf8"

#: BHSA's consonantal-only format (`g_cons_utf8` plus `trailer_utf8`), per its otext.tf.
_UNVOCALIZED_FORMAT = "text-orig-plain"


@dataclass(frozen=True, slots=True)
class SemanticPsalm:
    """One psalm's half-verse texts, in three variants, and their BHSA node ids."""

    number: int
    half_verse_nodes: tuple[int, ...] = ()
    half_verses: tuple[str, ...] = ()
    half_verses_unvocalized: tuple[str, ...] = ()
    half_verses_niqqud_only: tuple[str, ...] = ()


class Corpus(BaseCorpus[SemanticPsalm]):
    """A loaded BHSA Text-Fabric corpus, scoped to half-verse extraction."""

    @classmethod
    def load(
        cls, tf_path: Path | None = None, *, loader: Callable[..., Any] = shared_api
    ) -> Corpus:
        """Loads BHSA from `tf_path`, else $TEHILLIM_BHSA_PATH, else `DEFAULT_BHSA_CLONE`."""
        return cls(loader(tf_path, _REQUIRED_FEATURES))

    def _extract(self, number: int, half_verse_nodes: tuple[int, ...]) -> SemanticPsalm:
        """Renders each half-verse in the pointed, consonantal, and accent-stripped text tiers."""
        L, T = self._api.L, self._api.T  # noqa: N806
        half_verses = tuple(T.text(L.d(hv, otype="word")).strip() for hv in half_verse_nodes)
        return SemanticPsalm(
            number=number,
            half_verses=half_verses,
            half_verses_unvocalized=tuple(
                T.text(L.d(hv, otype="word"), fmt=_UNVOCALIZED_FORMAT).strip()
                for hv in half_verse_nodes
            ),
            half_verses_niqqud_only=tuple(strip_accents(half_verse) for half_verse in half_verses),
            half_verse_nodes=half_verse_nodes,
        )
