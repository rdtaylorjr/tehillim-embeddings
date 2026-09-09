"""Frozen, label-blind minimum-support thresholds for the clause vocabularies."""

from __future__ import annotations

__all__ = [
    "MIN_EXTERNAL_SUPPORT_K_CLAUSE_ATOM_TYP",
    "MIN_EXTERNAL_SUPPORT_K_CLAUSE_RELA",
    "MIN_EXTERNAL_SUPPORT_K_CLAUSE_SIGNATURE",
    "MIN_EXTERNAL_SUPPORT_K_CLAUSE_TYP",
]

# One rule sets all four: inherit 1000 where it holds Psalms coverage above 90 percent,
# else the largest measured curve point that does. See CLAUSE_LEVEL_PLAN.md for the curves.

# Frozen 2026-09-08. Inherited: 26 of 43 values, 91.5% coverage.
MIN_EXTERNAL_SUPPORT_K_CLAUSE_ATOM_TYP = 1000

# Frozen 2026-09-08. Inherited: 6 of 13 values, 96.7% coverage.
MIN_EXTERNAL_SUPPORT_K_CLAUSE_RELA = 1000

# Frozen 2026-09-08. Departure: clause-level typ holds only 80.5% at 1000, 95.1% here.
MIN_EXTERNAL_SUPPORT_K_CLAUSE_TYP = 500

# Frozen 2026-09-08. Departure: the joint typ:rela product holds only 63.7% at 1000, 91.2% here.
MIN_EXTERNAL_SUPPORT_K_CLAUSE_SIGNATURE = 100
