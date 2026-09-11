"""BHSA clause-atom relation code bands, with the provenance firewall parallelism scoring needs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "CODE_BANDS",
    "FORBIDDEN_FOR_PARALLELISM",
    "TARGET_ADJACENT",
    "Scope",
    "band_of",
    "band_vocabulary",
    "retained_codes",
]


@dataclass(frozen=True, slots=True)
class CodeBand:
    """One documented BHSA `code` range and the relation class it names."""

    name: str
    low: int
    high: int


#: The ranges in BHSA's own docs/features/code.md, in the order that file lists them.
CODE_BANDS: tuple[CodeBand, ...] = (
    CodeBand("no_relation", 0, 0),
    CodeBand("relative", 10, 16),
    CodeBand("infinitive_construct", 50, 74),
    CodeBand("asyndetic", 100, 167),
    CodeBand("parallel", 200, 201),
    CodeBand("defective", 220, 223),
    CodeBand("conjunctive_adverb", 300, 367),
    CodeBand("coordinate", 400, 487),
    CodeBand("postulational", 500, 567),
    CodeBand("conditional", 600, 667),
    CodeBand("temporal", 700, 767),
    CodeBand("final", 800, 867),
    CodeBand("causal", 900, 967),
    CodeBand("direct_speech", 999, 999),
)

#: BHSA defines 200 and 201 as parallel clause atoms, which is the parallelism target's construct.
FORBIDDEN_FOR_PARALLELISM = frozenset({"parallel"})

#: A defective atom shares its predicate with a mother or daughter, and gapping tracks parallelism.
TARGET_ADJACENT = frozenset({"defective"})


class Scope(Enum):
    """Which benchmark a clause representation is built for, and so what it may carry."""

    PARALLELISM = "parallelism"
    GENRE = "genre"


#: Excluding parallel and defective empties 472 of 5,203 colons (9.07%), so genre only.
SHIPPABLE_SCOPES: tuple[Scope, ...] = (Scope.GENRE,)


def _excluded(scope: Scope) -> frozenset[str]:
    """The bands this scope may not see."""
    if scope is Scope.PARALLELISM:
        return FORBIDDEN_FOR_PARALLELISM | TARGET_ADJACENT
    return frozenset()


def band_of(code: int) -> str:
    """The documented relation class of one BHSA clause-atom `code`."""
    for band in CODE_BANDS:
        if band.low <= code <= band.high:
            return band.name
    raise ValueError(f"clause-atom code {code} falls in no documented band")


def band_vocabulary(scope: Scope) -> tuple[str, ...]:
    """The band names a representation for this scope may carry, in stable dimension order."""
    excluded = _excluded(scope)
    return tuple(sorted(band.name for band in CODE_BANDS if band.name not in excluded))


def retained_codes(codes: tuple[int, ...], scope: Scope) -> tuple[int, ...]:
    """The codes this scope keeps, dropping excluded atoms so no mass deficit reports them."""
    excluded = _excluded(scope)
    return tuple(code for code in codes if band_of(code) not in excluded)
