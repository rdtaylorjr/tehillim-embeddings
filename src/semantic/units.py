"""The unit every semantic dataset is built from, and the one step that encodes units to rows."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from core.text import TextTier

#: Encodes texts to one row each, in order: a sentence-transformers model, a raw transformer with
#: last-token pooling, or a hosted API, all behind the same call.
type Encoder = Callable[[Sequence[str]], np.ndarray]


@dataclass(frozen=True, slots=True)
class Unit:
    """One row of a semantic dataset: the node it is keyed to and its text in each tier it has."""

    node: int
    texts: Mapping[TextTier, str]


def texts_at(units: Sequence[Unit], tier: TextTier) -> list[str]:
    """Every unit's text in one tier, refusing a unit the tier is absent from."""
    missing = [unit.node for unit in units if tier not in unit.texts]
    if missing:
        raise ValueError(f"{len(missing)} units have no {tier} text, first node {missing[0]}")
    return [unit.texts[tier] for unit in units]


def embed(units: Sequence[Unit], tier: TextTier, encode: Encoder) -> dict[int, np.ndarray]:
    """One vector per unit, keyed by node, from one call of the encoder over the tier's texts."""
    if not units:
        raise ValueError("no units to embed")
    vectors = np.asarray(encode(texts_at(units, tier)))
    if vectors.shape[0] != len(units):
        raise ValueError(f"encoder returned {vectors.shape[0]} rows for {len(units)} units")
    return {unit.node: vectors[index] for index, unit in enumerate(units)}
