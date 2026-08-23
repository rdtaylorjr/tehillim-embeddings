"""Loads BHSA phrase-atom type per half-verse for the Psalms via Text-Fabric."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tf.fabric import Fabric

DEFAULT_BHSA_TF_PATH = Path.home() / "Developer" / "hebrew" / "bhsa" / "tf" / "2021"

_REQUIRED_FEATURES = "otype book chapter verse typ function det rela"

_PSALMS_BOOK_NAME = "Psalmi"


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


class Corpus:
    """A loaded BHSA Text-Fabric corpus, scoped to half-verse phrase-atom type extraction."""

    def __init__(self, api: Any) -> None:
        self._api = api

    @property
    def api(self) -> Any:
        """The underlying Text-Fabric API, for whole-corpus queries outside Psalms."""
        return self._api

    @classmethod
    def load(cls, tf_path: Path | None = None) -> Corpus:
        """Loads BHSA from `tf_path`, or `DEFAULT_BHSA_TF_PATH`."""
        path = tf_path or DEFAULT_BHSA_TF_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"BHSA Text-Fabric data not found at {path}. "
                "Clone https://github.com/ETCBC/bhsa and pass its tf/<version> directory."
            )
        tf = Fabric(locations=[str(path)], silent="deep")
        api = tf.load(_REQUIRED_FEATURES, silent="deep")
        if api is None:
            raise RuntimeError(f"Text-Fabric failed to load required features from {path}")
        return cls(api)

    def psalms(self) -> list[PhrasePsalm]:
        """Extracts all 150 psalms' half-verse phrase-atom type sequences, in canonical order."""
        F, L, T = self._api.F, self._api.L, self._api.T  # noqa: N806

        book_nodes = [b for b in F.otype.s("book") if F.book.v(b) == _PSALMS_BOOK_NAME]
        if not book_nodes:
            raise RuntimeError(f"Book '{_PSALMS_BOOK_NAME}' not found in loaded corpus")

        psalms: list[PhrasePsalm] = []
        for chapter_node in L.d(book_nodes[0], otype="chapter"):
            _, psalm_number = T.sectionFromNode(chapter_node)
            half_verse_nodes = L.d(chapter_node, otype="half_verse")
            atoms_by_hv = [L.d(hv, otype="phrase_atom") for hv in half_verse_nodes]
            mothers_by_hv = [[L.u(pa, otype="phrase")[0] for pa in atoms] for atoms in atoms_by_hv]

            half_verse_typ = tuple(tuple(F.typ.v(pa) for pa in atoms) for atoms in atoms_by_hv)
            half_verse_det = tuple(tuple(F.det.v(pa) for pa in atoms) for atoms in atoms_by_hv)
            half_verse_rela = tuple(tuple(F.rela.v(pa) for pa in atoms) for atoms in atoms_by_hv)
            half_verse_function = tuple(
                tuple(F.function.v(mother) for mother in mothers) for mothers in mothers_by_hv
            )
            half_verse_n_words = tuple(
                tuple(len(L.d(pa, otype="word")) for pa in atoms) for atoms in atoms_by_hv
            )
            half_verse_phrase_id = tuple(tuple(mothers) for mothers in mothers_by_hv)
            half_verse_phrase_atom_count = tuple(
                tuple(len(L.d(mother, otype="phrase_atom")) for mother in mothers)
                for mothers in mothers_by_hv
            )
            half_verse_subphrase_rela = tuple(
                tuple(F.rela.v(sp) for sp in L.d(hv, otype="subphrase")) for hv in half_verse_nodes
            )

            psalms.append(
                PhrasePsalm(
                    number=psalm_number,
                    half_verse_nodes=tuple(half_verse_nodes),
                    half_verse_typ=half_verse_typ,
                    half_verse_function=half_verse_function,
                    half_verse_det=half_verse_det,
                    half_verse_rela=half_verse_rela,
                    half_verse_n_words=half_verse_n_words,
                    half_verse_phrase_id=half_verse_phrase_id,
                    half_verse_phrase_atom_count=half_verse_phrase_atom_count,
                    half_verse_subphrase_rela=half_verse_subphrase_rela,
                )
            )

        psalms.sort(key=lambda p: p.number)
        return psalms
