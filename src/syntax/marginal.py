"""`[phrase_typ; phrase_function]` marginal baseline for H5.5's conjunction-vs-marginals test."""

from __future__ import annotations

import numpy as np

from core.columns import PsalmColumns
from core.ngram import pooled_ngram_psalm_vectors
from syntax.corpus import PhrasePsalm
from syntax.function_ngram import phrase_function_unigram_histogram
from syntax.typ_ngram import phrase_typ_unigram_histogram
from syntax.vocabulary import FUNCTION_VOCABULARY, TYP_VOCABULARY


def typ_function_marginal_vectors(psalms: list[PhrasePsalm]) -> dict[int, np.ndarray]:
    """`[phrase_typ_1gram; phrase_function_1gram]` per node, independent marginal histograms."""
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, half_verse_typ, half_verse_function in zip(
            psalm.half_verse_nodes, psalm.half_verse_typ, psalm.half_verse_function, strict=True
        ):
            vectors[node] = np.concatenate(
                [
                    phrase_typ_unigram_histogram(half_verse_typ),
                    phrase_function_unigram_histogram(half_verse_function),
                ]
            )
    return vectors


def typ_function_marginal_psalm_vectors(psalms: list[PhrasePsalm]) -> dict[int, np.ndarray]:
    """Psalm-broadcast `typ_function_marginal_vectors`."""
    typ_columns = [PsalmColumns(p.number, p.half_verse_nodes, p.half_verse_typ) for p in psalms]
    function_columns = [
        PsalmColumns(p.number, p.half_verse_nodes, p.half_verse_function) for p in psalms
    ]
    typ_vectors = pooled_ngram_psalm_vectors(typ_columns, (1,), TYP_VOCABULARY, order_by_node=None)
    function_vectors = pooled_ngram_psalm_vectors(
        function_columns, (1,), FUNCTION_VOCABULARY, order_by_node=None
    )
    return {
        node: np.concatenate([typ_vectors[node], function_vectors[node]]) for node in typ_vectors
    }
