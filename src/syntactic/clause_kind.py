"""Normalized clause-kind inventory per colon: the coarsest clause representation, stage 6A."""

from __future__ import annotations

from functools import partial

from core.ngram import ngram_psalm_vectors, ngram_vectors
from syntactic.assignment import majority_mask
from syntactic.clause_columns import colon_sequences
from syntactic.clause_vocabulary import KIND_VOCABULARY
from syntactic.corpus import ClausePsalm

__all__ = [
    "clause_kind_1gram_psalm_vectors",
    "clause_kind_1gram_vectors",
    "clause_kind_columns",
]

_INDEX_OF = {value: index for index, value in enumerate(KIND_VOCABULARY)}
_DIM = len(KIND_VOCABULARY)


def clause_kind_columns(psalm: ClausePsalm) -> tuple[tuple[str, ...], ...]:
    """One clause-kind sequence per colon, under the majority assignment rule."""
    return colon_sequences(
        psalm.clause_kind,
        psalm.clause_assignment,
        len(psalm.half_verse_nodes),
        mask=majority_mask(psalm.clause_assignment),
    )


clause_kind_1gram_vectors = partial(
    ngram_vectors, columns_of=clause_kind_columns, vocabulary=KIND_VOCABULARY, orders=(1,)
)
clause_kind_1gram_psalm_vectors = partial(
    ngram_psalm_vectors, columns_of=clause_kind_columns, vocabulary=KIND_VOCABULARY, orders=(1,)
)
clause_kind_1_2gram_vectors = partial(
    ngram_vectors, columns_of=clause_kind_columns, vocabulary=KIND_VOCABULARY, orders=(1, 2)
)
clause_kind_1_2gram_psalm_vectors = partial(
    ngram_psalm_vectors, columns_of=clause_kind_columns, vocabulary=KIND_VOCABULARY, orders=(1, 2)
)
clause_kind_1_2_3gram_vectors = partial(
    ngram_vectors, columns_of=clause_kind_columns, vocabulary=KIND_VOCABULARY, orders=(1, 2, 3)
)
clause_kind_1_2_3gram_psalm_vectors = partial(
    ngram_psalm_vectors,
    columns_of=clause_kind_columns,
    vocabulary=KIND_VOCABULARY,
    orders=(1, 2, 3),
)
