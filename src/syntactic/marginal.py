"""`[phrase_typ; phrase_function]` marginal baseline for H5.5's conjunction-vs-marginals test."""

from __future__ import annotations

import numpy as np

from core.columns import PsalmColumns
from core.ngram import pooled_ngram_psalm_vectors, unigram_histogram
from core.vocabulary import index_map
from syntactic.corpus import PhrasePsalm
from syntactic.vocabulary import FUNCTION_VOCABULARY, TYP_VOCABULARY

_TYP_INDEX = index_map(TYP_VOCABULARY)
_FUNCTION_INDEX = index_map(FUNCTION_VOCABULARY)


def typ_function_marginal_vectors(psalms: list[PhrasePsalm]) -> dict[int, np.ndarray]:
    """`[phrase_typ_1gram; phrase_function_1gram]` per node, independent marginal histograms."""
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, half_verse_typ, half_verse_function in zip(
            psalm.half_verse_nodes, psalm.half_verse_typ, psalm.half_verse_function, strict=True
        ):
            vectors[node] = np.concatenate(
                [
                    unigram_histogram(half_verse_typ, _TYP_INDEX, len(TYP_VOCABULARY)),
                    unigram_histogram(
                        half_verse_function, _FUNCTION_INDEX, len(FUNCTION_VOCABULARY)
                    ),
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
