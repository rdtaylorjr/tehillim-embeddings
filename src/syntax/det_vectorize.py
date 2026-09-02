"""Normalized determination (det) unigram histogram per half-verse: H5.8's independent baseline."""

from __future__ import annotations

import numpy as np

from core.ngram import pooled_ngram_psalm_vectors, unigram_histogram
from syntax.corpus import PhrasePsalm
from syntax.vocabulary import DET_VOCABULARY

_INDEX_OF = {value: i for i, value in enumerate(DET_VOCABULARY)}
_DIM = len(DET_VOCABULARY)


def phrase_det_unigram_histogram(half_verse_det: tuple[str, ...]) -> np.ndarray:
    """Normalized determined/undetermined/NA proportions over one half-verse: count(det) / m."""
    return unigram_histogram(half_verse_det, _INDEX_OF, _DIM)


def phrase_det_1gram_vectors(psalms: list[PhrasePsalm]) -> dict[int, np.ndarray]:
    """One `phrase_det_1gram` histogram per half-verse node."""
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, half_verse_det in zip(psalm.half_verse_nodes, psalm.half_verse_det, strict=True):
            vectors[node] = phrase_det_unigram_histogram(half_verse_det)
    return vectors


def phrase_det_1gram_psalm_vectors(psalms: list[PhrasePsalm]) -> dict[int, np.ndarray]:
    """Psalm-broadcast `phrase_det_1gram`: atom-count-weighted pooling across every half-verse."""
    columns = [(p.half_verse_nodes, p.half_verse_det) for p in psalms]
    return pooled_ngram_psalm_vectors(columns, (1,), _INDEX_OF, _DIM, order_by_node=None)
