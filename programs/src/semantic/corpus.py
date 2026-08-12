"""Loads Hebrew Psalms half-verse text and BHSA node ids via Text-Fabric."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tf.fabric import Fabric

DEFAULT_BHSA_TF_PATH = Path.home() / "Developer" / "hebrew" / "bhsa" / "tf" / "2021"

_REQUIRED_FEATURES = "otype book chapter verse g_word_utf8 g_cons_utf8 trailer_utf8"

#: BHSA's registered format for consonantal-only text (`g_cons_utf8` plus
#: `trailer_utf8`), confirmed against BHSA's own otext.tf.
_UNVOCALIZED_FORMAT = "text-orig-plain"

_PSALMS_BOOK_NAME = "Psalmi"

#: Unicode range of Hebrew cantillation marks (U+0591 through U+05AF).
#: Niqqud starts at U+05B0, immediately after this range.
_ACCENT_RANGE = range(0x0591, 0x05B0)


def _strip_accents(text: str) -> str:
    """Removes cantillation marks from vocalized text, keeping niqqud."""
    return "".join(ch for ch in text if ord(ch) not in _ACCENT_RANGE)


@dataclass(frozen=True, slots=True)
class Psalm:
    """One psalm's half-verse texts, in three variants, and their BHSA node ids."""

    number: int
    half_verses: tuple[str, ...] = ()
    half_verses_unvocalized: tuple[str, ...] = ()
    half_verses_niqqud_only: tuple[str, ...] = ()
    half_verse_nodes: tuple[int, ...] = ()


class Corpus:
    """A loaded BHSA Text-Fabric corpus, scoped to half-verse extraction."""

    def __init__(self, api: Any) -> None:
        self._api = api

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

    def psalms(self) -> list[Psalm]:
        """Extracts all 150 psalms, in canonical order."""
        F, L, T = self._api.F, self._api.L, self._api.T  # noqa: N806

        book_nodes = [b for b in F.otype.s("book") if F.book.v(b) == _PSALMS_BOOK_NAME]
        if not book_nodes:
            raise RuntimeError(f"Book '{_PSALMS_BOOK_NAME}' not found in loaded corpus")

        psalms: list[Psalm] = []
        for chapter_node in L.d(book_nodes[0], otype="chapter"):
            _, psalm_number = T.sectionFromNode(chapter_node)
            half_verse_nodes = L.d(chapter_node, otype="half_verse")
            half_verses = tuple(T.text(L.d(hv, otype="word")).strip() for hv in half_verse_nodes)
            half_verses_unvocalized = tuple(
                T.text(L.d(hv, otype="word"), fmt=_UNVOCALIZED_FORMAT).strip()
                for hv in half_verse_nodes
            )
            half_verses_niqqud_only = tuple(_strip_accents(hv) for hv in half_verses)

            psalms.append(
                Psalm(
                    number=psalm_number,
                    half_verses=half_verses,
                    half_verses_unvocalized=half_verses_unvocalized,
                    half_verses_niqqud_only=half_verses_niqqud_only,
                    half_verse_nodes=tuple(half_verse_nodes),
                )
            )

        psalms.sort(key=lambda p: p.number)
        return psalms
