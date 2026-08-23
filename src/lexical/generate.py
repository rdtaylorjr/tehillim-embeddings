"""Computes and writes lexical datasets as Parquet: 19 weightings each for homograph and lexeme."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from lexical.corpus import Corpus, LexicalPsalm
from lexical.export import dataset_path, write_dataset
from lexical.frequency import icf_weights as compute_icf_weights
from lexical.frequency import lex0_token_frequencies, lex_token_frequencies, total_token_count
from lexical.positional import positional_icf_vectors
from lexical.psalm_position import psalm_positional_icf_vectors
from lexical.psalm_recurrence import psalm_spacing_profile_vectors
from lexical.psalm_zoning import psalm_position_mean_vectors
from lexical.recurrence import spacing_profile_vectors
from lexical.vectorize import (
    binary_presence_vectors,
    icf_weighted_vectors,
    log_count_vectors,
    term_frequency_vectors,
    tf_icf_vectors,
)
from lexical.vocabulary import VocabularyKey, build_vocabulary
from lexical.zoning import position_mean_vectors

_FULL_WEIGHTS = (
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
)
_VOCAB_NAMES: dict[VocabularyKey, str] = {"lex0": "homograph", "lex": "lexeme"}
_POSITIONAL_K = {"icf_position2": 2, "icf_position4": 4, "icf_position8": 8}
_SPACING_K = {"icf_spacing2": 2, "icf_spacing4": 4, "icf_spacing8": 8}
_PSALM_POSITIONAL_K = {"icf_position2_psalm": 2, "icf_position4_psalm": 4, "icf_position8_psalm": 8}
_PSALM_SPACING_K = {"icf_spacing2_psalm": 2, "icf_spacing4_psalm": 4, "icf_spacing8_psalm": 8}


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
    if weight in _SPACING_K:
        return spacing_profile_vectors(psalms, vocabulary, key, icf_weights, k=_SPACING_K[weight])
    if weight == "icf_position_mean":
        return position_mean_vectors(psalms, vocabulary, key, icf_weights)
    if weight in _PSALM_POSITIONAL_K:
        return psalm_positional_icf_vectors(
            psalms, vocabulary, key, icf_weights, k=_PSALM_POSITIONAL_K[weight]
        )
    if weight in _PSALM_SPACING_K:
        return psalm_spacing_profile_vectors(
            psalms, vocabulary, key, icf_weights, k=_PSALM_SPACING_K[weight]
        )
    if weight == "icf_position_mean_psalm":
        return psalm_position_mean_vectors(psalms, vocabulary, key, icf_weights)
    raise ValueError(f"unknown weight {weight!r}")


def generate(
    psalms: list[LexicalPsalm],
    output_root: Path,
    icf_weights_by_key: dict[VocabularyKey, dict[str, float]],
) -> list[str]:
    """Writes every not-yet-written (vocab, weight) dataset, returns the names written."""
    plan: list[tuple[VocabularyKey, str]] = [
        (key, weight) for key in ("lex0", "lex") for weight in _FULL_WEIGHTS
    ]

    written: list[str] = []
    for key, weight in plan:
        vocab_name = _VOCAB_NAMES[key]
        if dataset_path(output_root, vocab_name, weight).exists():
            continue
        print(f"computing lexical unit={vocab_name} construction={weight}...", file=sys.stderr)
        vocabulary = build_vocabulary(psalms, key=key)
        vectors = _vectors_for_weight(psalms, vocabulary, key, weight, icf_weights_by_key[key])
        description = (
            f"Lexical vectors over the {vocab_name} unit (BHSA {key} feature), "
            f"construction={weight}, dimension {len(vocabulary)}."
        )
        write_dataset(output_root, vocab_name, weight, vectors, description)
        written.append(f"{vocab_name}_{weight}")
    return written


def main() -> None:
    """Generates every missing lexical dataset: 19 weightings each for homograph and lexeme."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    output_root = parser.parse_args().output_root
    corpus = Corpus.load()
    psalms = corpus.psalms()
    total_tokens = total_token_count(corpus.api)
    icf_weights_by_key: dict[VocabularyKey, dict[str, float]] = {
        "lex0": compute_icf_weights(lex0_token_frequencies(corpus.api), total_tokens),
        "lex": compute_icf_weights(lex_token_frequencies(corpus.api), total_tokens),
    }
    written = generate(psalms, output_root, icf_weights_by_key)
    print(f"wrote {len(written)} dataset files", file=sys.stderr)


if __name__ == "__main__":
    main()
