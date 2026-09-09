"""Loads BHSA phrase-atom type per half-verse for the Psalms via Text-Fabric."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.corpus import DEFAULT_BHSA_CLONE, BaseCorpus, load_api

__all__ = ["DEFAULT_BHSA_CLONE", "Corpus", "PhrasePsalm"]

_REQUIRED_FEATURES = "otype book chapter verse typ function det rela"


@dataclass(frozen=True, slots=True)
class PhrasePsalm:
    """One psalm's half-verse phrase-atom feature sequences, aligned atom-for-atom."""

    number: int
    half_verse_nodes: tuple[int, ...] = ()
    half_verse_typ: tuple[tuple[str, ...], ...] = ()
    half_verse_function: tuple[tuple[str, ...], ...] = ()
    half_verse_det: tuple[tuple[str, ...], ...] = ()
    half_verse_rela: tuple[tuple[str, ...], ...] = ()
    half_verse_n_words: tuple[tuple[int, ...], ...] = ()
    half_verse_phrase_id: tuple[tuple[int, ...], ...] = ()
    half_verse_phrase_atom_count: tuple[tuple[int, ...], ...] = ()
    half_verse_subphrase_rela: tuple[tuple[str, ...], ...] = ()


class Corpus(BaseCorpus[PhrasePsalm]):
    """A loaded BHSA Text-Fabric corpus, scoped to half-verse phrase-atom type extraction."""

    @classmethod
    def load(cls, tf_path: Path | None = None, *, loader: Callable[..., Any] = load_api) -> Corpus:
        """Loads BHSA from `tf_path`, else $TEHILLIM_BHSA_PATH, else `DEFAULT_BHSA_CLONE`."""
        return cls(loader(tf_path, _REQUIRED_FEATURES))

    def _extract(self, number: int, half_verse_nodes: tuple[int, ...]) -> PhrasePsalm:
        """Reads each half-verse's phrase atoms and the mother phrases they belong to."""
        F, L = self._api.F, self._api.L  # noqa: N806
        atoms_by_hv = [L.d(hv, otype="phrase_atom") for hv in half_verse_nodes]
        mothers_by_hv = [[L.u(pa, otype="phrase")[0] for pa in atoms] for atoms in atoms_by_hv]
        return PhrasePsalm(
            number=number,
            half_verse_nodes=half_verse_nodes,
            half_verse_typ=tuple(tuple(F.typ.v(pa) for pa in atoms) for atoms in atoms_by_hv),
            half_verse_function=tuple(
                tuple(F.function.v(mother) for mother in mothers) for mothers in mothers_by_hv
            ),
            half_verse_det=tuple(tuple(F.det.v(pa) for pa in atoms) for atoms in atoms_by_hv),
            half_verse_rela=tuple(tuple(F.rela.v(pa) for pa in atoms) for atoms in atoms_by_hv),
            half_verse_n_words=tuple(
                tuple(len(L.d(pa, otype="word")) for pa in atoms) for atoms in atoms_by_hv
            ),
            half_verse_phrase_id=tuple(tuple(mothers) for mothers in mothers_by_hv),
            half_verse_phrase_atom_count=tuple(
                tuple(len(L.d(mother, otype="phrase_atom")) for mother in mothers)
                for mothers in mothers_by_hv
            ),
            half_verse_subphrase_rela=tuple(
                tuple(F.rela.v(sp) for sp in L.d(hv, otype="subphrase")) for hv in half_verse_nodes
            ),
        )
