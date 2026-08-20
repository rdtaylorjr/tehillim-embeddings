"""Computes and writes lexical datasets as Parquet: 19 lex0 weightings, and frozen lex_binary."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from lexical.corpus import Corpus, LexicalPsalm
from lexical.export import dataset_path, write_dataset
from lexical.frequency import icf_weights as compute_icf_weights
from lexical.frequency import lex0_token_frequencies, total_token_count
from lexical.positional import positional_icf_vectors
from lexical.psalm_position import psalm_positional_icf_vectors
from lexical.psalm_recurrence import psalm_lag_profile_vectors
from lexical.psalm_zoning import psalm_positional_centroid_vectors
from lexical.recurrence import lag_profile_vectors
from lexical.vectorize import (
    binary_presence_vectors,
    icf_weighted_vectors,
    log_count_vectors,
    term_frequency_vectors,
    tf_icf_vectors,
)
from lexical.vocabulary import VocabularyKey, build_vocabulary
from lexical.zoning import positional_centroid_vectors

_LEX0_WEIGHTS = (
    "binary",
    "count",
    "log_count",
    "icf",
    "tf_icf",
    "icf_pos2",
    "icf_pos4",
    "icf_pos8",
    "icf_lag2",
    "icf_lag4",
    "icf_lag8",
    "icf_posmean",
    "icf_pos2_psalm",
    "icf_pos4_psalm",
    "icf_pos8_psalm",
    "icf_lag2_psalm",
    "icf_lag4_psalm",
    "icf_lag8_psalm",
    "icf_posmean_psalm",
)
_LEX_WEIGHTS = ("binary",)
_POSITIONAL_K = {"icf_pos2": 2, "icf_pos4": 4, "icf_pos8": 8}
_LAG_K = {"icf_lag2": 2, "icf_lag4": 4, "icf_lag8": 8}
_PSALM_POSITIONAL_K = {"icf_pos2_psalm": 2, "icf_pos4_psalm": 4, "icf_pos8_psalm": 8}
_PSALM_LAG_K = {"icf_lag2_psalm": 2, "icf_lag4_psalm": 4, "icf_lag8_psalm": 8}


def _vectors_for_weight(
    psalms: list[LexicalPsalm],
    vocabulary: tuple[str, ...],
    key: VocabularyKey,
    weight: str,
    icf_weights: dict[str, float],
) -> dict[int, np.ndarray]:
    """Dispatches to the vector-building function matching `weight`."""
    if weight == "binary":
        return binary_presence_vectors(psalms, vocabulary, key)
    if weight == "count":
        return term_frequency_vectors(psalms, vocabulary, key)
    if weight == "log_count":
        return log_count_vectors(psalms, vocabulary, key)
    if weight == "icf":
        return icf_weighted_vectors(psalms, vocabulary, key, icf_weights)
    if weight == "tf_icf":
        return tf_icf_vectors(psalms, vocabulary, key, icf_weights)
    if weight in _POSITIONAL_K:
        return positional_icf_vectors(psalms, vocabulary, key, icf_weights, k=_POSITIONAL_K[weight])
    if weight in _LAG_K:
        return lag_profile_vectors(psalms, vocabulary, key, icf_weights, k=_LAG_K[weight])
    if weight == "icf_posmean":
        return positional_centroid_vectors(psalms, vocabulary, key, icf_weights)
    if weight in _PSALM_POSITIONAL_K:
        return psalm_positional_icf_vectors(
            psalms, vocabulary, key, icf_weights, k=_PSALM_POSITIONAL_K[weight]
        )
    if weight in _PSALM_LAG_K:
        return psalm_lag_profile_vectors(
            psalms, vocabulary, key, icf_weights, k=_PSALM_LAG_K[weight]
        )
    if weight == "icf_posmean_psalm":
        return psalm_positional_centroid_vectors(psalms, vocabulary, key, icf_weights)
    raise ValueError(f"unknown weight {weight!r}")


def generate(
    psalms: list[LexicalPsalm], output_root: Path, icf_weights: dict[str, float]
) -> list[str]:
    """Writes every not-yet-written (vocab, weight) dataset, returns the names written."""
    plan: list[tuple[VocabularyKey, str]] = [("lex0", weight) for weight in _LEX0_WEIGHTS] + [
        ("lex", weight) for weight in _LEX_WEIGHTS
    ]

    written: list[str] = []
    for key, weight in plan:
        if dataset_path(output_root, key, weight).exists():
            continue
        print(f"computing lexical vocab={key} weight={weight}...", file=sys.stderr)
        vocabulary = build_vocabulary(psalms, key=key)
        vectors = _vectors_for_weight(psalms, vocabulary, key, weight, icf_weights)
        description = (
            f"Lexical vectors over the {key} vocabulary (BHSA feature), "
            f"weight={weight}, dimension {len(vocabulary)}."
        )
        write_dataset(output_root, key, weight, vectors, description)
        written.append(f"{key}_{weight}")
    return written


def main() -> None:
    """Generates every missing lexical dataset: 19 lex0 weightings plus the frozen lex_binary."""
    output_root = Path(__file__).resolve().parents[2]
    corpus = Corpus.load()
    psalms = corpus.psalms()
    lex0_frequencies = lex0_token_frequencies(corpus.api)
    total_tokens = total_token_count(corpus.api)
    icf_lookup = compute_icf_weights(lex0_frequencies, total_tokens)
    written = generate(psalms, output_root, icf_lookup)
    print(f"wrote {len(written)} dataset files", file=sys.stderr)


if __name__ == "__main__":
    main()
