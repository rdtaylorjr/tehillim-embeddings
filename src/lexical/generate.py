"""Computes and writes lexical datasets as Parquet: 5 form weightings, and frozen lexeme_binary."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from lexical.corpus import Corpus, LexicalPsalm
from lexical.export import dataset_path, write_dataset
from lexical.frequency import icf_weights as compute_icf_weights
from lexical.frequency import lex0_token_frequencies, total_token_count
from lexical.positional import positional_icf_vectors
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

_FORM_WEIGHTS = (
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
)
_LEXEME_WEIGHTS = ("binary",)
_POSITIONAL_K = {"icf_pos2": 2, "icf_pos4": 4, "icf_pos8": 8}
_LAG_K = {"icf_lag2": 2, "icf_lag4": 4, "icf_lag8": 8}


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
    raise ValueError(f"unknown weight {weight!r}")


def generate(
    psalms: list[LexicalPsalm], output_root: Path, icf_weights: dict[str, float]
) -> list[str]:
    """Writes every not-yet-written (vocab, weight) dataset, returns the names written."""
    plan: list[tuple[str, VocabularyKey, str]] = [
        ("form", "lex0", weight) for weight in _FORM_WEIGHTS
    ] + [("lexeme", "lex", weight) for weight in _LEXEME_WEIGHTS]

    written: list[str] = []
    for vocab_name, key, weight in plan:
        if dataset_path(output_root, vocab_name, weight).exists():
            continue
        print(f"computing lexical vocab={vocab_name} weight={weight}...", file=sys.stderr)
        vocabulary = build_vocabulary(psalms, key=key)
        vectors = _vectors_for_weight(psalms, vocabulary, key, weight, icf_weights)
        description = (
            f"Lexical vectors over the {vocab_name} vocabulary ({key} BHSA feature), "
            f"weight={weight}, dimension {len(vocabulary)}."
        )
        write_dataset(output_root, vocab_name, weight, vectors, description)
        written.append(f"{vocab_name}_{weight}")
    return written


def main() -> None:
    """Generates every missing lexical dataset: 8 form weightings plus the frozen lexeme_binary."""
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
