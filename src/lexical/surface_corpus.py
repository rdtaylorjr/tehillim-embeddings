"""Loads BHSA surface word-form features per half-verse, in three text tiers, via Text-Fabric."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tf.fabric import Fabric

DEFAULT_BHSA_TF_PATH = Path.home() / "Developer" / "hebrew" / "bhsa" / "tf" / "2021"

_REQUIRED_FEATURES = "otype book chapter verse g_cons_utf8 g_word_utf8"

_PSALMS_BOOK_NAME = "Psalmi"

#: Unicode range of Hebrew cantillation marks (U+0591 through U+05AF); niqqud starts at U+05B0.
_ACCENT_RANGE = range(0x0591, 0x05B0)


def _strip_accents(text: str) -> str:
    """Removes cantillation marks from a word's pointed form, keeping niqqud."""
    return "".join(ch for ch in text if ord(ch) not in _ACCENT_RANGE)


@dataclass(frozen=True, slots=True)
class SurfacePsalm:
    """One psalm's half-verse surface word-form sequences, in three text tiers, aligned words."""

    number: int
    half_verse_consonantal: tuple[tuple[str, ...], ...] = ()
    half_verse_vocalized: tuple[tuple[str, ...], ...] = ()
    half_verse_cantillation: tuple[tuple[str, ...], ...] = ()
    half_verse_nodes: tuple[int, ...] = ()


class SurfaceCorpus:
    """A loaded BHSA Text-Fabric corpus, scoped to half-verse surface word-form extraction."""

    def __init__(self, api: Any) -> None:
        self._api = api

    @property
    def api(self) -> Any:
        """The underlying Text-Fabric API, for whole-corpus queries outside Psalms."""
        return self._api

    @classmethod
    def load(cls, tf_path: Path | None = None) -> SurfaceCorpus:
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

    def psalms(self) -> list[SurfacePsalm]:
        """Extracts all 150 psalms' half-verse surface-form sequences, in canonical order."""
        F, L, T = self._api.F, self._api.L, self._api.T  # noqa: N806

        book_nodes = [b for b in F.otype.s("book") if F.book.v(b) == _PSALMS_BOOK_NAME]
        if not book_nodes:
            raise RuntimeError(f"Book '{_PSALMS_BOOK_NAME}' not found in loaded corpus")

        psalms: list[SurfacePsalm] = []
        for chapter_node in L.d(book_nodes[0], otype="chapter"):
            _, psalm_number = T.sectionFromNode(chapter_node)
            half_verse_nodes = L.d(chapter_node, otype="half_verse")
            half_verse_consonantal = tuple(
                tuple(F.g_cons_utf8.v(w) for w in L.d(hv, otype="word")) for hv in half_verse_nodes
            )
            half_verse_cantillation = tuple(
                tuple(F.g_word_utf8.v(w) for w in L.d(hv, otype="word")) for hv in half_verse_nodes
            )
            half_verse_vocalized = tuple(
                tuple(_strip_accents(F.g_word_utf8.v(w)) for w in L.d(hv, otype="word"))
                for hv in half_verse_nodes
            )

            psalms.append(
                SurfacePsalm(
                    number=psalm_number,
                    half_verse_consonantal=half_verse_consonantal,
                    half_verse_vocalized=half_verse_vocalized,
                    half_verse_cantillation=half_verse_cantillation,
                    half_verse_nodes=tuple(half_verse_nodes),
                )
            )

        psalms.sort(key=lambda p: p.number)
        return psalms
