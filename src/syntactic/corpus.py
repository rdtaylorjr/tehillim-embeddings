"""Loads BHSA phrase- and clause-level syntactic features per half-verse for the Psalms."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from core.corpus import DEFAULT_BHSA_CLONE, BaseCorpus, load_api
from syntactic.assignment import Assignment, assign_nodes_to_units

__all__ = [
    "CLAUSE",
    "DEFAULT_BHSA_CLONE",
    "PHRASE",
    "ClausePsalm",
    "Corpus",
    "LevelExtractor",
    "PhrasePsalm",
    "clause_corpus",
    "phrase_corpus",
]

_PHRASE_FEATURES = "otype book chapter verse typ function det rela"
_CLAUSE_FEATURES = "otype book chapter verse typ kind rela code tab is_root dist dist_unit mother"


def _empty_assignment() -> Assignment:
    """An assignment covering nothing, for a psalm record built without half-verses."""
    empty = np.empty(0, dtype=np.int64)
    return Assignment(node_index=empty, unit_index=empty, weight=np.empty(0, dtype=np.float64))


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


@dataclass(frozen=True, slots=True)
class ClausePsalm:
    """One psalm's clauses and clause atoms, with the word fraction each puts in every colon."""

    number: int
    half_verse_nodes: tuple[int, ...] = ()
    clause_nodes: tuple[int, ...] = ()
    clause_typ: tuple[str, ...] = ()
    clause_kind: tuple[str, ...] = ()
    clause_rela: tuple[str, ...] = ()
    clause_dist: tuple[int, ...] = ()
    clause_dist_unit: tuple[str, ...] = ()
    clause_assignment: Assignment = field(default_factory=_empty_assignment)
    clause_atom_nodes: tuple[int, ...] = ()
    clause_atom_typ: tuple[str, ...] = ()
    clause_atom_code: tuple[int, ...] = ()
    clause_atom_tab: tuple[int, ...] = ()
    clause_atom_is_root: tuple[str, ...] = ()
    clause_atom_mother: tuple[int | None, ...] = ()
    clause_atom_assignment: Assignment = field(default_factory=_empty_assignment)


class LevelExtractor[RecordT](Protocol):
    """Builds one syntactic level's psalm record from the half-verses of a psalm."""

    @property
    def required_features(self) -> str:
        """The Text-Fabric features this level reads."""

    def extract(self, api: Any, number: int, half_verse_nodes: tuple[int, ...]) -> RecordT:
        """One psalm's record for this level."""
        ...


def _overlaps(
    api: Any, otype: str, unit_nodes: tuple[int, ...]
) -> tuple[tuple[int, ...], Assignment]:
    """Every `otype` node touching these colons, with the word fraction it puts in each."""
    E, L = api.E, api.L  # noqa: N806
    unit_slots = [np.asarray(E.oslots.s(unit), dtype=np.int64) for unit in unit_nodes]
    unit_flat = np.concatenate(unit_slots)
    #: Descending the psalm once, as the phrase extractor does, rather than ascending per slot.
    nodes = tuple(L.d(L.u(unit_nodes[0], otype="chapter")[0], otype=otype))
    node_slots = [np.asarray(E.oslots.s(node), dtype=np.int64) for node in nodes]
    node_flat = np.concatenate(node_slots)
    #: A node can reach past this psalm's colons, so the frame has to span both slot sets.
    low = int(min(unit_flat.min(), node_flat.min()))
    high = int(max(unit_flat.max(), node_flat.max()))
    slot_unit = np.full(high - low + 1, -1, dtype=np.int64)
    slot_unit[unit_flat - low] = np.repeat(
        np.arange(len(unit_nodes), dtype=np.int64), [slots.size for slots in unit_slots]
    )
    counts = np.asarray([slots.size for slots in node_slots], dtype=np.int64)
    return nodes, assign_nodes_to_units(slot_unit, node_flat - low, counts, n_units=len(unit_nodes))


class PhraseExtractor:
    """Reads each half-verse's phrase atoms and the mother phrases they belong to."""

    @property
    def required_features(self) -> str:
        """The Text-Fabric features phrase-level extraction reads."""
        return _PHRASE_FEATURES

    def extract(self, api: Any, number: int, half_verse_nodes: tuple[int, ...]) -> PhrasePsalm:
        """One psalm's phrase-atom record."""
        F, L = api.F, api.L  # noqa: N806
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


class ClauseExtractor:
    """Reads the clauses and clause atoms overlapping each colon, weighted by word fraction."""

    @property
    def required_features(self) -> str:
        """The Text-Fabric features clause-level extraction reads."""
        return _CLAUSE_FEATURES

    def extract(self, api: Any, number: int, half_verse_nodes: tuple[int, ...]) -> ClausePsalm:
        """One psalm's clause and clause-atom record."""
        if not half_verse_nodes:
            return ClausePsalm(number=number)
        F, E = api.F, api.E  # noqa: N806
        clauses, clause_assignment = _overlaps(api, "clause", half_verse_nodes)
        atoms, atom_assignment = _overlaps(api, "clause_atom", half_verse_nodes)
        return ClausePsalm(
            number=number,
            half_verse_nodes=half_verse_nodes,
            clause_nodes=clauses,
            clause_typ=tuple(F.typ.v(c) for c in clauses),
            clause_kind=tuple(F.kind.v(c) for c in clauses),
            clause_rela=tuple(F.rela.v(c) for c in clauses),
            clause_dist=tuple(F.dist.v(c) for c in clauses),
            clause_dist_unit=tuple(F.dist_unit.v(c) for c in clauses),
            clause_assignment=clause_assignment,
            clause_atom_nodes=atoms,
            clause_atom_typ=tuple(F.typ.v(a) for a in atoms),
            clause_atom_code=tuple(F.code.v(a) for a in atoms),
            clause_atom_tab=tuple(F.tab.v(a) for a in atoms),
            clause_atom_is_root=tuple(F.is_root.v(a) for a in atoms),
            clause_atom_mother=tuple(E.mother.f(a)[0] if E.mother.f(a) else None for a in atoms),
            clause_atom_assignment=atom_assignment,
        )


PHRASE: LevelExtractor[PhrasePsalm] = PhraseExtractor()
CLAUSE: LevelExtractor[ClausePsalm] = ClauseExtractor()


class Corpus[RecordT](BaseCorpus[RecordT]):
    """A loaded BHSA Text-Fabric corpus, scoped to one syntactic level's extraction."""

    def __init__(self, api: Any, extractor: LevelExtractor[RecordT]) -> None:
        """Wraps an already-loaded Text-Fabric API and the level it is read at."""
        super().__init__(api)
        self._extractor = extractor

    @classmethod
    def load(
        cls,
        extractor: LevelExtractor[RecordT],
        tf_path: Path | None = None,
        *,
        loader: Callable[..., Any] = load_api,
    ) -> Corpus[RecordT]:
        """Loads BHSA from `tf_path`, else $TEHILLIM_BHSA_PATH, else `DEFAULT_BHSA_CLONE`."""
        return cls(loader(tf_path, extractor.required_features), extractor)

    @classmethod
    def sharing(cls, api: Any, extractor: LevelExtractor[RecordT]) -> Corpus[RecordT]:
        """Reads a second level from an already-loaded API rather than loading BHSA twice."""
        return cls(api, extractor)

    def _extract(self, number: int, half_verse_nodes: tuple[int, ...]) -> RecordT:
        """Delegates to this corpus's level extractor."""
        return self._extractor.extract(self._api, number, half_verse_nodes)


def phrase_corpus(tf_path: Path | None = None) -> Corpus[PhrasePsalm]:
    """A corpus that reads the Psalms at phrase-atom level."""
    return Corpus.load(PHRASE, tf_path)


def clause_corpus(tf_path: Path | None = None) -> Corpus[ClausePsalm]:
    """A corpus that reads the Psalms at clause and clause-atom level."""
    return Corpus.load(CLAUSE, tf_path)
