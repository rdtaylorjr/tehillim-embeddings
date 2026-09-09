"""The per-psalm view every lexical vectorizer needs: node ids and their value sequences."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PsalmColumns:
    """One psalm's number, half-verse node ids, and their per-half-verse value sequences."""

    number: int
    nodes: tuple[int, ...]
    half_verses: tuple[tuple[str, ...], ...]
