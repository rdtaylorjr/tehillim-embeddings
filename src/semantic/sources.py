"""The corpora the semantic domain embeds: each a scope of the tree, its tiers, and its units."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from core.partition import BHSA, BHSA_HALF_VERSE, DSS, Scope
from core.text import TextTier
from semantic.registry import VARIATIONS
from semantic.scroll import RECONSTRUCTIONS, VERSE, ScrollCorpus
from semantic.units import Unit


@dataclass(frozen=True, slots=True)
class Source:
    """One corpus at one unit: where its rows sit, which tiers it renders, how it is read."""

    scope: Scope
    tiers: tuple[TextTier, ...]
    load: Callable[[], Sequence[Unit]]

    @property
    def slug(self) -> str:
        """The scope's values joined, naming the cell that writes this source."""
        return "-".join(value for _, value in self.scope.values())


def _bhsa_units(unit: str) -> Callable[[], Sequence[Unit]]:
    """A loader of the Masoretic Psalms at one unit, opened when a cell runs, never at planning."""

    def load() -> Sequence[Unit]:
        from semantic.corpus import Corpus

        return Corpus.load(unit=unit).units()

    return load


def _scroll_units(witness: str, reconstruction: str) -> Callable[[], Sequence[Unit]]:
    """A loader of one scroll's verse units under one reconstruction."""
    return lambda: ScrollCorpus.load(witness, reconstruction).units()


#: The scrolls read from the dss corpus, each named as its `scroll` feature names it.
WITNESSES: tuple[str, ...] = ("11Q5",)

#: The BHSA node types the Psalms are embedded at: the accentual half-verse and the verse.
BHSA_UNITS: tuple[str, ...] = (BHSA_HALF_VERSE.unit, VERSE)

SOURCES: tuple[Source, ...] = (
    *(
        Source(Scope(BHSA, unit), tuple(tier for tier, _ in VARIATIONS), _bhsa_units(unit))
        for unit in BHSA_UNITS
    ),
    *(
        Source(
            Scope(DSS, VERSE, witness=witness, reconstruction=reconstruction),
            ("consonantal",),
            _scroll_units(witness, reconstruction),
        )
        for witness in WITNESSES
        for reconstruction in RECONSTRUCTIONS
    ),
)


def source_for(scope: Scope, sources: Sequence[Source] = SOURCES) -> Source:
    """The source that writes one scope, refusing a scope no source declares."""
    for source in sources:
        if source.scope == scope:
            return source
    raise KeyError(f"no semantic source for {scope.directory}")


def scope_arguments(scope: Scope) -> tuple[str, ...]:
    """The command-line flags that select a scope, one per key it carries."""
    return tuple(f"--{key}={value}" for key, value in scope.values())
