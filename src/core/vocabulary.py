"""Vocabulary-to-column indexing, shared by every histogram builder."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def index_map(vocabulary: Sequence[str]) -> dict[str, int]:
    """Maps each vocabulary value to its column index, defining histogram column order."""
    return {value: i for i, value in enumerate(vocabulary)}


def sorted_distinct(sequences: Iterable[Sequence[str]]) -> tuple[str, ...]:
    """Every distinct value across the given sequences, in sorted order."""
    return tuple(sorted({value for sequence in sequences for value in sequence}))
