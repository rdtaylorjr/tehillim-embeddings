"""Per-clause signatures: the joint (typ, rela) conjunction, e.g. `NmCl:Subj`."""

from __future__ import annotations

__all__ = ["build_clause_signature", "clause_signatures"]


def build_clause_signature(*, typ: str, rela: str) -> str:
    """`typ:rela`-style signature, e.g. `NmCl:Subj`; NA is kept, being the modal relation."""
    return f"{typ}:{rela}"


def clause_signatures(*, typ: tuple[str, ...], rela: tuple[str, ...]) -> tuple[str, ...]:
    """One signature per clause, aligned across the two per-clause feature sequences."""
    return tuple(
        build_clause_signature(typ=clause_typ, rela=clause_rela)
        for clause_typ, clause_rela in zip(typ, rela, strict=True)
    )
