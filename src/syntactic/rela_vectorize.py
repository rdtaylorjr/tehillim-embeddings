"""Normalized safe-relation unigram histogram per half-verse: `Para` masked before counting."""

from __future__ import annotations

from functools import partial

from core.ngram import ngram_psalm_vectors, ngram_vectors
from syntactic.rela import SAFE_RELA_VOCABULARY, safe_rela_columns

phrase_rela_1gram_vectors = partial(
    ngram_vectors, columns_of=safe_rela_columns, vocabulary=SAFE_RELA_VOCABULARY, orders=(1,)
)
phrase_rela_1gram_psalm_vectors = partial(
    ngram_psalm_vectors, columns_of=safe_rela_columns, vocabulary=SAFE_RELA_VOCABULARY, orders=(1,)
)

phrase_rela_1_2gram_vectors = partial(
    ngram_vectors, columns_of=safe_rela_columns, vocabulary=SAFE_RELA_VOCABULARY, orders=(1, 2)
)
phrase_rela_1_2gram_psalm_vectors = partial(
    ngram_psalm_vectors,
    columns_of=safe_rela_columns,
    vocabulary=SAFE_RELA_VOCABULARY,
    orders=(1, 2),
)
phrase_rela_1_2_3gram_vectors = partial(
    ngram_vectors, columns_of=safe_rela_columns, vocabulary=SAFE_RELA_VOCABULARY, orders=(1, 2, 3)
)
phrase_rela_1_2_3gram_psalm_vectors = partial(
    ngram_psalm_vectors,
    columns_of=safe_rela_columns,
    vocabulary=SAFE_RELA_VOCABULARY,
    orders=(1, 2, 3),
)
