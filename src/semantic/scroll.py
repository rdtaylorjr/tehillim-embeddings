"""Loads one Judaean Desert scroll as verse units from ETCBC/dss and the tehillim-scribes module."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.corpus import load_locations
from core.partition import DSS, Scope
from core.text import consonantal
from semantic.units import Unit

DEFAULT_DSS_PATH = Path.home() / "Developer" / "hebrew" / "etcbc" / "dss" / "tf" / "2.0"
DSS_PATH_ENV = "TEHILLIM_DSS_PATH"
DEFAULT_SCRIBES_PATH = Path.home() / "Developer" / "tehillim" / "tehillim-scribes" / "tf" / "2.0"
SCRIBES_PATH_ENV = "TEHILLIM_SCRIBES_TF_PATH"

#: The dss features that place a word, and the scribes features that give it its text and grades.
_REQUIRED_FEATURES = (
    "otype scroll book chapter verse biblical type "
    "composition composition_verse gword gw_cons_utf8 grade"
)

#: The dss value of `book` on the Psalms, the one book the collation covers.
PSALMS_BOOK = "Ps"

#: Which of a graphic word's letters enter the text, by the scribes `grade` of each letter:
#: `c` certain, `p` probable, `q` possible, `x` an unidentifiable trace, `r` reconstructed,
#: `m` a missing sign. `none` takes the extant letters and no reconstruction.
RECONSTRUCTIONS: Mapping[str, frozenset[str]] = {"none": frozenset("cpq")}

#: The unit the scroll is cut at: the verse of the psalm, or of the composition, the word stands in.
VERSE = "verse"

#: Shin and sin dots ride on the letter before them and take that letter's grade.
_COMBINING_MARKS = frozenset({"ׁ", "ׂ"})


def dss_location(env: Mapping[str, str] | None = None) -> Path:
    """The local dss directory: $TEHILLIM_DSS_PATH when set, else the conventional clone."""
    environment = os.environ if env is None else env
    return Path(environment.get(DSS_PATH_ENV) or DEFAULT_DSS_PATH)


def scribes_location(env: Mapping[str, str] | None = None) -> Path:
    """The local scribes module: $TEHILLIM_SCRIBES_TF_PATH when set, else the checkout's tf/2.0."""
    environment = os.environ if env is None else env
    return Path(environment.get(SCRIBES_PATH_ENV) or DEFAULT_SCRIBES_PATH)


def load_api(
    dss: Path | None = None,
    scribes: Path | None = None,
    *,
    load: Callable[..., Any] = load_locations,
    env: Mapping[str, str] | None = None,
) -> Any:
    """Loads dss and the scribes module together, so every feature reads off one node space."""
    locations = [dss or dss_location(env), scribes or scribes_location(env)]
    api = load(locations, _REQUIRED_FEATURES)
    if api is None:
        raise RuntimeError(
            f"Text-Fabric could not load dss with the scribes module from {locations}"
        )
    return api


#: What a word belongs to: (book or composition, chapter or None, verse), or None outside both.
type Label = tuple[str, str | None, str]


def label_of(F: Any, node: int) -> Label | None:  # noqa: N803 -- Text-Fabric's own name
    """The verse a word stands in: the psalm's from dss, else the composition's from scribes."""
    if F.biblical.v(node) and F.book.v(node) == PSALMS_BOOK:
        return (PSALMS_BOOK, str(F.chapter.v(node)), str(F.verse.v(node)))
    composition, verse = F.composition.v(node), F.composition_verse.v(node)
    if composition is not None and verse is not None:
        return (str(composition), None, str(verse))
    return None


def kept_letters(word: str, grade: str, keep: frozenset[str]) -> str:
    """The letters of a graphic word whose grade is kept, each with the marks that ride on it."""
    letters = [ch for ch in word if ch not in _COMBINING_MARKS]
    if len(letters) != len(grade):
        raise ValueError(f"{word!r} has {len(letters)} letters but {len(grade)} grades {grade!r}")
    out: list[str] = []
    index = -1
    for ch in word:
        if ch not in _COMBINING_MARKS:
            index += 1
        if grade[index] in keep:
            out.append(ch)
    return "".join(out)


@dataclass(frozen=True, slots=True)
class Run:
    """A maximal run of consecutive words under one label, in leather order."""

    label: Label
    nodes: tuple[int, ...]


def runs(F: Any, words: Sequence[int]) -> Iterator[Run]:  # noqa: N803
    """Cuts the scroll's words into labelled runs, dropping words that stand in no verse."""
    current: Label | None = None
    nodes: list[int] = []
    for node in words:
        label = label_of(F, node)
        if label != current:
            if current is not None:
                yield Run(current, tuple(nodes))
            current, nodes = label, []
        if label is not None:
            nodes.append(node)
    if current is not None:
        yield Run(current, tuple(nodes))


def text_of(F: Any, run: Run, keep: frozenset[str]) -> str:  # noqa: N803
    """The run's graphic words, once each, the letters the reconstruction keeps, as consonants."""
    tokens = []
    for node in run.nodes:
        word, grade = F.gw_cons_utf8.v(node), F.grade.v(node)
        #: The graphic word is written on every node it spans, so only its head node emits it.
        if word is None or F.gword.v(node) != node:
            continue
        kept = kept_letters(word, grade, keep)
        if kept:
            tokens.append(kept)
    return consonantal(" ".join(tokens))


@dataclass(frozen=True, slots=True)
class ScrollCorpus:
    """One witness of the dss corpus, read at the verse under one reconstruction."""

    api: Any
    witness: str
    reconstruction: str = "none"

    @classmethod
    def load(
        cls,
        witness: str,
        reconstruction: str = "none",
        *,
        loader: Callable[..., Any] = load_api,
    ) -> ScrollCorpus:
        """Opens dss with the scribes module and scopes it to one scroll."""
        return cls(loader(), witness, reconstruction)

    @property
    def scope(self) -> Scope:
        """Where this corpus's rows sit in the tree."""
        return Scope(DSS, VERSE, witness=self.witness, reconstruction=self.reconstruction)

    def words(self) -> list[int]:
        """Every word node of the scroll in leather order, punctuation left out."""
        F, L = self.api.F, self.api.L  # noqa: N806
        scrolls = [n for n in F.otype.s("scroll") if F.scroll.v(n) == self.witness]
        if not scrolls:
            raise LookupError(f"{self.witness} is not a scroll of the dss corpus")
        return [
            word
            for scroll in scrolls
            for word in L.d(scroll, otype="word")
            if F.type.v(word) != "punct"
        ]

    def units(self) -> list[Unit]:
        """One unit per verse run that keeps any text, keyed by the run's first word node."""
        keep = RECONSTRUCTIONS[self.reconstruction]
        F = self.api.F  # noqa: N806
        units = []
        for run in runs(F, self.words()):
            text = text_of(F, run, keep)
            if text:
                units.append(Unit(run.nodes[0], {"consonantal": text}))
        return units
