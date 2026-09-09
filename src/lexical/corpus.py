"""Loads BHSA lexical features (lex, lex0) per half-verse for the Hebrew Psalms via Text-Fabric."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.corpus import DEFAULT_BHSA_CLONE, BaseCorpus, load_api

__all__ = ["DEFAULT_BHSA_CLONE", "Corpus", "LexicalPsalm"]

_REQUIRED_FEATURES = "otype book chapter verse lex lex0 freq_lex"


@dataclass(frozen=True, slots=True)
class LexicalPsalm:
    """One psalm's half-verse lex and lex0 sequences, aligned word-for-word, and their node ids."""

    number: int
    half_verse_nodes: tuple[int, ...] = ()
    half_verse_lexemes: tuple[tuple[str, ...], ...] = ()
    half_verse_forms: tuple[tuple[str, ...], ...] = ()


class Corpus(BaseCorpus[LexicalPsalm]):
    """A loaded BHSA Text-Fabric corpus, scoped to half-verse lexical feature extraction."""

    @classmethod
    def load(cls, tf_path: Path | None = None, *, loader: Callable[..., Any] = load_api) -> Corpus:
        """Loads BHSA from `tf_path`, else $TEHILLIM_BHSA_PATH, else `DEFAULT_BHSA_CLONE`."""
        return cls(loader(tf_path, _REQUIRED_FEATURES))

    def _extract(self, number: int, half_verse_nodes: tuple[int, ...]) -> LexicalPsalm:
        """Reads each half-verse's lex and lex0 word sequences."""
        F, L = self._api.F, self._api.L  # noqa: N806
        words_by_half_verse = [L.d(hv, otype="word") for hv in half_verse_nodes]
        return LexicalPsalm(
            number=number,
            half_verse_lexemes=tuple(
                tuple(F.lex.v(w) for w in words) for words in words_by_half_verse
            ),
            half_verse_forms=tuple(
                tuple(F.lex0.v(w) for w in words) for words in words_by_half_verse
            ),
            half_verse_nodes=half_verse_nodes,
        )
