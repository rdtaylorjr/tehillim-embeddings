"""Vocabulary-to-column indexing, shared by every histogram builder."""

from __future__ import annotations

from collections.abc import Sequence


def index_map(vocabulary: Sequence[str]) -> dict[str, int]:
    """Maps each vocabulary value to its column index, defining histogram column order."""
    return {value: i for i, value in enumerate(vocabulary)}
