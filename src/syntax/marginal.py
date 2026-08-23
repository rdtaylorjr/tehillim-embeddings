"""`[phrase_typ; phrase_function]` marginal baseline for H5.5's conjunction-vs-marginals test."""

from __future__ import annotations

import numpy as np

from morphology.ngram import pooled_ngram_psalm_vectors
from syntax.corpus import PhrasePsalm
from syntax.function_ngram import _DIM as _FUNCTION_DIM
from syntax.function_ngram import _INDEX_OF as _FUNCTION_INDEX_OF
from syntax.function_ngram import phrase_function_unigram_histogram
from syntax.typ_ngram import _DIM as _TYP_DIM
from syntax.typ_ngram import _INDEX_OF as _TYP_INDEX_OF
from syntax.typ_ngram import phrase_typ_unigram_histogram


def typ_function_marginal_vectors(psalms: list[PhrasePsalm]) -> dict[int, np.ndarray]:
    """`[phrase_typ_1gram; phrase_function_1gram]` per colon, independent marginal histograms."""
    vectors: dict[int, np.ndarray] = {}
    for psalm in psalms:
        for node, colon_typ, colon_function in zip(
            psalm.half_verse_nodes, psalm.half_verse_typ, psalm.half_verse_function, strict=True
        ):
            vectors[node] = np.concatenate(
                [
                    phrase_typ_unigram_histogram(colon_typ),
                    phrase_function_unigram_histogram(colon_function),
                ]
            )
    return vectors


def typ_function_marginal_psalm_vectors(psalms: list[PhrasePsalm]) -> dict[int, np.ndarray]:
    """Psalm-broadcast `typ_function_marginal_vectors`."""
    typ_columns = [(p.half_verse_nodes, p.half_verse_typ) for p in psalms]
    function_columns = [(p.half_verse_nodes, p.half_verse_function) for p in psalms]
    typ_vectors = pooled_ngram_psalm_vectors(
        typ_columns, (1,), _TYP_INDEX_OF, _TYP_DIM, order_by_node=None
    )
    function_vectors = pooled_ngram_psalm_vectors(
        function_columns, (1,), _FUNCTION_INDEX_OF, _FUNCTION_DIM, order_by_node=None
    )
    return {
        node: np.concatenate([typ_vectors[node], function_vectors[node]]) for node in typ_vectors
    }
