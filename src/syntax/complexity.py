"""Conventional structural-complexity summaries per colon: not new inferential machinery."""

from __future__ import annotations

import numpy as np

from syntax.corpus import PhrasePsalm

_N_FEATURES = 4


def colon_complexity_features(
    n_words: tuple[int, ...], phrase_id: tuple[int, ...], phrase_atom_count: tuple[int, ...]
) -> np.ndarray:
    """`[n_atoms, n_phrases, mean_words_per_atom, proportion_multi_atom_phrases]` for one colon."""
    n_atoms = len(n_words)
    if n_atoms == 0:
        return np.zeros(_N_FEATURES, dtype=np.float32)

    distinct_phrases = {}
    for pid, count in zip(phrase_id, phrase_atom_count, strict=True):
        distinct_phrases[pid] = count
    n_phrases = len(distinct_phrases)
    mean_words_per_atom = float(np.mean(n_words))
    proportion_multi_atom = sum(1 for count in distinct_phrases.values() if count > 1) / n_phrases

    return np.array(
        [n_atoms, n_phrases, mean_words_per_atom, proportion_multi_atom], dtype=np.float32
    )


def phrase_complexity_vectors(psalms: list[PhrasePsalm]) -> dict[int, np.ndarray]:
    """One complexity feature vector per colon node."""
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, n_words, phrase_id, phrase_atom_count in zip(
            psalm.colon_nodes,
            psalm.colon_n_words,
            psalm.colon_phrase_id,
            psalm.colon_phrase_atom_count,
            strict=True,
        ):
            vectors[node] = colon_complexity_features(n_words, phrase_id, phrase_atom_count)
    return vectors


def phrase_complexity_psalm_vectors(psalms: list[PhrasePsalm]) -> dict[int, np.ndarray]:
    """Psalm-broadcast complexity vector: unweighted mean of every colon's own features."""
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        colon_features = [
            colon_complexity_features(n_words, phrase_id, phrase_atom_count)
            for n_words, phrase_id, phrase_atom_count in zip(
                psalm.colon_n_words,
                psalm.colon_phrase_id,
                psalm.colon_phrase_atom_count,
                strict=True,
            )
        ]
        psalm_vector = (
            np.mean(colon_features, axis=0)
            if colon_features
            else np.zeros(_N_FEATURES, dtype=np.float32)
        )
        for node in psalm.colon_nodes:
            vectors[node] = psalm_vector.astype(np.float32)
    return vectors
