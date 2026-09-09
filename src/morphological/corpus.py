"""Loads BHSA word-level morphology features per half-verse for the Psalms via Text-Fabric."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.corpus import DEFAULT_BHSA_CLONE, BaseCorpus, load_api

__all__ = ["DEFAULT_BHSA_CLONE", "Corpus", "MorphologicalPsalm"]

_REQUIRED_FEATURES = "otype book chapter verse sp gn nu ps st vs vt prs_gn prs_nu prs_ps"

_MORPHOLOGY_FEATURES = ("sp", "gn", "nu", "ps", "st", "vs", "vt", "prs_gn", "prs_nu", "prs_ps")


@dataclass(frozen=True, slots=True)
class MorphologicalPsalm:
    """One psalm's half-verse morphology feature sequences, aligned word-for-word, and node ids."""

    number: int
    half_verse_nodes: tuple[int, ...] = ()
    half_verse_sp: tuple[tuple[str, ...], ...] = ()
    half_verse_gn: tuple[tuple[str, ...], ...] = ()
    half_verse_nu: tuple[tuple[str, ...], ...] = ()
    half_verse_ps: tuple[tuple[str, ...], ...] = ()
    half_verse_st: tuple[tuple[str, ...], ...] = ()
    half_verse_vs: tuple[tuple[str, ...], ...] = ()
    half_verse_vt: tuple[tuple[str, ...], ...] = ()
    half_verse_prs_gn: tuple[tuple[str, ...], ...] = ()
    half_verse_prs_nu: tuple[tuple[str, ...], ...] = ()
    half_verse_prs_ps: tuple[tuple[str, ...], ...] = ()


class Corpus(BaseCorpus[MorphologicalPsalm]):
    """A loaded BHSA Text-Fabric corpus, scoped to half-verse morphology feature extraction."""

    @classmethod
    def load(cls, tf_path: Path | None = None, *, loader: Callable[..., Any] = load_api) -> Corpus:
        """Loads BHSA from `tf_path`, else $TEHILLIM_BHSA_PATH, else `DEFAULT_BHSA_CLONE`."""
        return cls(loader(tf_path, _REQUIRED_FEATURES))

    def _extract(self, number: int, half_verse_nodes: tuple[int, ...]) -> MorphologicalPsalm:
        """Reads every tracked morphology feature's word sequence for each half-verse."""
        F, L = self._api.F, self._api.L  # noqa: N806
        words_by_half_verse = [L.d(hv, otype="word") for hv in half_verse_nodes]
        half_verse_by_feature = {
            feature: tuple(
                tuple(getattr(F, feature).v(w) for w in words) for words in words_by_half_verse
            )
            for feature in _MORPHOLOGY_FEATURES
        }
        return MorphologicalPsalm(
            number=number,
            half_verse_nodes=half_verse_nodes,
            **{
                f"half_verse_{feature}": values for feature, values in half_verse_by_feature.items()
            },
        )
