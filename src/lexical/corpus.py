"""Loads BHSA lexical features (lex, lex0) per half-verse for the Hebrew Psalms via Text-Fabric."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tf.fabric import Fabric

DEFAULT_BHSA_TF_PATH = Path.home() / "Developer" / "hebrew" / "bhsa" / "tf" / "2021"

_REQUIRED_FEATURES = "otype book chapter verse lex lex0"

_PSALMS_BOOK_NAME = "Psalmi"


@dataclass(frozen=True, slots=True)
class LexicalPsalm:
    """One psalm's half-verse lex and lex0 sequences, aligned word-for-word, and their node ids."""

    number: int
    half_verse_lexemes: tuple[tuple[str, ...], ...] = ()
    half_verse_forms: tuple[tuple[str, ...], ...] = ()
    half_verse_nodes: tuple[int, ...] = ()


class Corpus:
    """A loaded BHSA Text-Fabric corpus, scoped to half-verse lexical feature extraction."""

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

    def psalms(self) -> list[LexicalPsalm]:
        """Extracts all 150 psalms' half-verse lex/lex0 sequences, in canonical order."""
        F, L, T = self._api.F, self._api.L, self._api.T  # noqa: N806

        book_nodes = [b for b in F.otype.s("book") if F.book.v(b) == _PSALMS_BOOK_NAME]
        if not book_nodes:
            raise RuntimeError(f"Book '{_PSALMS_BOOK_NAME}' not found in loaded corpus")

        psalms: list[LexicalPsalm] = []
        for chapter_node in L.d(book_nodes[0], otype="chapter"):
            _, psalm_number = T.sectionFromNode(chapter_node)
            half_verse_nodes = L.d(chapter_node, otype="half_verse")
            half_verse_lexemes = tuple(
                tuple(F.lex.v(w) for w in L.d(hv, otype="word")) for hv in half_verse_nodes
            )
            half_verse_forms = tuple(
                tuple(F.lex0.v(w) for w in L.d(hv, otype="word")) for hv in half_verse_nodes
            )

            psalms.append(
                LexicalPsalm(
                    number=psalm_number,
                    half_verse_lexemes=half_verse_lexemes,
                    half_verse_forms=half_verse_forms,
                    half_verse_nodes=tuple(half_verse_nodes),
                )
            )

        psalms.sort(key=lambda p: p.number)
        return psalms
