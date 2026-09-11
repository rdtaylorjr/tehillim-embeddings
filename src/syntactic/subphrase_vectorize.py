"""Normalized safe subphrase-relation histogram per half-verse: `par` masked before counting."""

from __future__ import annotations

from functools import partial

from core.ngram import ngram_psalm_vectors, ngram_vectors
from syntactic.subphrase import SAFE_SUBPHRASE_RELA_VOCABULARY, safe_subphrase_rela_columns

subphrase_rela_1gram_vectors = partial(
    ngram_vectors,
    columns_of=safe_subphrase_rela_columns,
    vocabulary=SAFE_SUBPHRASE_RELA_VOCABULARY,
    orders=(1,),
)
subphrase_rela_1gram_psalm_vectors = partial(
    ngram_psalm_vectors,
    columns_of=safe_subphrase_rela_columns,
    vocabulary=SAFE_SUBPHRASE_RELA_VOCABULARY,
    orders=(1,),
)

subphrase_rela_1_2gram_vectors = partial(
    ngram_vectors,
    columns_of=safe_subphrase_rela_columns,
    vocabulary=SAFE_SUBPHRASE_RELA_VOCABULARY,
    orders=(1, 2),
)
subphrase_rela_1_2gram_psalm_vectors = partial(
    ngram_psalm_vectors,
    columns_of=safe_subphrase_rela_columns,
    vocabulary=SAFE_SUBPHRASE_RELA_VOCABULARY,
    orders=(1, 2),
)
subphrase_rela_1_2_3gram_vectors = partial(
    ngram_vectors,
    columns_of=safe_subphrase_rela_columns,
    vocabulary=SAFE_SUBPHRASE_RELA_VOCABULARY,
    orders=(1, 2, 3),
)
subphrase_rela_1_2_3gram_psalm_vectors = partial(
    ngram_psalm_vectors,
    columns_of=safe_subphrase_rela_columns,
    vocabulary=SAFE_SUBPHRASE_RELA_VOCABULARY,
    orders=(1, 2, 3),
)
