"""Loads the Psalms from the BHSA as units at one node type, one text per tier, keyed by node."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.corpus import DEFAULT_BHSA_CLONE, BaseCorpus, shared_api
from core.partition import HALF_VERSE
from core.text import consonantal, strip_accents
from semantic.units import Unit

__all__ = ["DEFAULT_BHSA_CLONE", "Corpus"]

_REQUIRED_FEATURES = "otype book chapter verse g_word_utf8 g_cons_utf8 trailer_utf8"

#: BHSA's unpointed format (`g_cons_utf8` plus `trailer_utf8`), per its otext.tf, which still
#: carries maqaf, paseq, sof pasuq and the shin and sin dots that `consonantal` removes.
_UNVOCALIZED_FORMAT = "text-orig-plain"


class Corpus(BaseCorpus[list[Unit]]):
    """A loaded BHSA Text-Fabric corpus, scoped to one unit's texts in the three tiers."""

    @classmethod
    def load(
        cls,
        tf_path: Path | None = None,
        *,
        unit: str = HALF_VERSE,
        loader: Callable[..., Any] = shared_api,
    ) -> Corpus:
        """Loads BHSA from `tf_path`, else $TEHILLIM_BHSA_PATH, else `DEFAULT_BHSA_CLONE`."""
        return cls(loader(tf_path, _REQUIRED_FEATURES), unit)

    def _extract(self, number: int, half_verse_nodes: tuple[int, ...]) -> list[Unit]:
        """Renders each half-verse in the pointed, consonantal, and accent-stripped text tiers."""
        del number  # a unit is keyed by its node, the psalm number places nothing
        L, T = self._api.L, self._api.T  # noqa: N806
        units = []
        for node in half_verse_nodes:
            words = L.d(node, otype="word")
            cantillation = T.text(words).strip()
            units.append(
                Unit(
                    node,
                    {
                        "consonantal": consonantal(T.text(words, fmt=_UNVOCALIZED_FORMAT)),
                        "vocalized": strip_accents(cantillation),
                        "cantillation": cantillation,
                    },
                )
            )
        return units

    def units(self) -> list[Unit]:
        """Every unit of the 150 psalms, in canonical order."""
        return [unit for psalm in self.psalms() for unit in psalm]
