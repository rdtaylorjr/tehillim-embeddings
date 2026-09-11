"""The 19 lexical constructions and the vector builder each one names, over any column view."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from core.columns import PsalmColumns
from lexical.positional import positional_icf_vectors
from lexical.psalm_position import psalm_positional_icf_vectors
from lexical.psalm_recurrence import psalm_spacing_profile_vectors
from lexical.psalm_vectorize import (
    psalm_binary_presence_vectors,
    psalm_icf_weighted_vectors,
    psalm_log_count_vectors,
    psalm_term_frequency_vectors,
    psalm_tf_icf_vectors,
)
from lexical.psalm_zoning import psalm_position_mean_vectors
from lexical.recurrence import spacing_profile_vectors
from lexical.vectorize import (
    binary_presence_vectors,
    icf_weighted_vectors,
    log_count_vectors,
    term_frequency_vectors,
    tf_icf_vectors,
)
from lexical.zoning import position_mean_vectors

VectorMap = dict[int, np.ndarray]

KBuilder = Callable[[list[PsalmColumns], tuple[str, ...], dict[str, float], int], VectorMap]


def partial_k(
    builder: KBuilder, k: int
) -> Callable[[list[PsalmColumns], tuple[str, ...], dict[str, float]], VectorMap]:
    """Binds the region count `k` for the construction families parameterized by it."""
    return lambda columns, vocabulary, icf: builder(columns, vocabulary, icf, k)


FULL_WEIGHTS = (
    "binary",
    "count",
    "log_count",
    "icf",
    "tf_icf",
    "icf_position2",
    "icf_position4",
    "icf_position8",
    "icf_spacing2",
    "icf_spacing4",
    "icf_spacing8",
    "icf_position_mean",
    "icf_position2_psalm",
    "icf_position4_psalm",
    "icf_position8_psalm",
    "icf_spacing2_psalm",
    "icf_spacing4_psalm",
    "icf_spacing8_psalm",
    "icf_position_mean_psalm",
    "binary_psalm",
    "count_psalm",
    "log_count_psalm",
    "icf_psalm",
    "tf_icf_psalm",
)

_POSITIONAL_K = {"icf_position2": 2, "icf_position4": 4, "icf_position8": 8}
_SPACING_K = {"icf_spacing2": 2, "icf_spacing4": 4, "icf_spacing8": 8}
_PSALM_POSITIONAL_K = {"icf_position2_psalm": 2, "icf_position4_psalm": 4, "icf_position8_psalm": 8}
_PSALM_SPACING_K = {"icf_spacing2_psalm": 2, "icf_spacing4_psalm": 4, "icf_spacing8_psalm": 8}


#: construction -> the builder that produces it, given (columns, vocabulary, icf_weights).
_BUILDERS: dict[
    str, Callable[[list[PsalmColumns], tuple[str, ...], dict[str, float]], VectorMap]
] = {
    "binary": lambda columns, vocabulary, _icf: binary_presence_vectors(columns, vocabulary),
    "count": lambda columns, vocabulary, _icf: term_frequency_vectors(columns, vocabulary),
    "log_count": lambda columns, vocabulary, _icf: log_count_vectors(columns, vocabulary),
    "icf": icf_weighted_vectors,
    "tf_icf": tf_icf_vectors,
    "binary_psalm": lambda columns, vocabulary, _icf: psalm_binary_presence_vectors(
        columns, vocabulary
    ),
    "count_psalm": lambda columns, vocabulary, _icf: psalm_term_frequency_vectors(
        columns, vocabulary
    ),
    "log_count_psalm": lambda columns, vocabulary, _icf: psalm_log_count_vectors(
        columns, vocabulary
    ),
    "icf_psalm": psalm_icf_weighted_vectors,
    "tf_icf_psalm": psalm_tf_icf_vectors,
    "icf_position_mean": position_mean_vectors,
    "icf_position_mean_psalm": psalm_position_mean_vectors,
    **{
        name: partial_k(positional_icf_vectors, k)
        for name, k in (("icf_position2", 2), ("icf_position4", 4), ("icf_position8", 8))
    },
    **{
        name: partial_k(spacing_profile_vectors, k)
        for name, k in (("icf_spacing2", 2), ("icf_spacing4", 4), ("icf_spacing8", 8))
    },
    **{
        name: partial_k(psalm_positional_icf_vectors, k)
        for name, k in (
            ("icf_position2_psalm", 2),
            ("icf_position4_psalm", 4),
            ("icf_position8_psalm", 8),
        )
    },
    **{
        name: partial_k(psalm_spacing_profile_vectors, k)
        for name, k in (
            ("icf_spacing2_psalm", 2),
            ("icf_spacing4_psalm", 4),
            ("icf_spacing8_psalm", 8),
        )
    },
}


def vectors_for_weight(
    columns: list[PsalmColumns],
    vocabulary: tuple[str, ...],
    weight: str,
    icf_weights: dict[str, float],
) -> VectorMap:
    """Dispatches to the vector-building function matching `weight`."""
    builder = _BUILDERS.get(weight)
    if builder is None:
        raise ValueError(f"unknown weight {weight!r}")
    return builder(columns, vocabulary, icf_weights)
