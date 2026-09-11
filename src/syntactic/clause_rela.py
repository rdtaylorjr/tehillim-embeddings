"""Clause-relation provenance: the values parallelism scoring may not see, and why."""

from __future__ import annotations

from syntactic.clause_code import Scope

__all__ = ["TARGET_ADJACENT_RELA", "retained_rela_indices", "safe_rela_vocabulary"]

#: Audited 2026-09-08: 7 of 1,598 Logos groups carry both, null mean 2.07 (sd 1.38), p=0.0035.
TARGET_ADJACENT_RELA = frozenset({"Resu", "ReVo"})


def _excluded(scope: Scope) -> frozenset[str]:
    """The relations this scope may not see."""
    return TARGET_ADJACENT_RELA if scope is Scope.PARALLELISM else frozenset()


def safe_rela_vocabulary(vocabulary: tuple[str, ...], scope: Scope) -> tuple[str, ...]:
    """`vocabulary` without the relations this scope excludes, in its original order."""
    excluded = _excluded(scope)
    return tuple(value for value in vocabulary if value not in excluded)


def retained_rela_indices(rela: tuple[str, ...], scope: Scope) -> tuple[int, ...]:
    """Positions this scope keeps, dropping excluded clauses so no mass deficit reports them."""
    excluded = _excluded(scope)
    return tuple(index for index, value in enumerate(rela) if value not in excluded)
