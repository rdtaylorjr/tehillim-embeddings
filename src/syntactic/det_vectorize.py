"""Normalized determination (det) unigram histogram per half-verse: H5.8's independent baseline."""

from __future__ import annotations

from functools import partial

from core.ngram import ngram_psalm_vectors, ngram_vectors
from syntactic.vocabulary import DET_VOCABULARY, det_columns

phrase_det_1gram_vectors = partial(
    ngram_vectors, columns_of=det_columns, vocabulary=DET_VOCABULARY, orders=(1,)
)
phrase_det_1gram_psalm_vectors = partial(
    ngram_psalm_vectors, columns_of=det_columns, vocabulary=DET_VOCABULARY, orders=(1,)
)

phrase_det_1_2gram_vectors = partial(
    ngram_vectors, columns_of=det_columns, vocabulary=DET_VOCABULARY, orders=(1, 2)
)
phrase_det_1_2gram_psalm_vectors = partial(
    ngram_psalm_vectors, columns_of=det_columns, vocabulary=DET_VOCABULARY, orders=(1, 2)
)
phrase_det_1_2_3gram_vectors = partial(
    ngram_vectors, columns_of=det_columns, vocabulary=DET_VOCABULARY, orders=(1, 2, 3)
)
phrase_det_1_2_3gram_psalm_vectors = partial(
    ngram_psalm_vectors, columns_of=det_columns, vocabulary=DET_VOCABULARY, orders=(1, 2, 3)
)
