from __future__ import annotations

import numpy as np

from core.ngram import unigram_histogram
from core.vocabulary import index_map
from syntactic.corpus import PhrasePsalm
from syntactic.marginal import (
    typ_function_marginal_psalm_vectors,
    typ_function_marginal_vectors,
)
from syntactic.vocabulary import FUNCTION_VOCABULARY, TYP_VOCABULARY

_TYP_INDEX = index_map(TYP_VOCABULARY)
_FUNCTION_INDEX = index_map(FUNCTION_VOCABULARY)
_TYP_DIM = len(TYP_VOCABULARY)
_FUNCTION_DIM = len(FUNCTION_VOCABULARY)


def _psalm(*, number, nodes, typ, function):
    return PhrasePsalm(
        number=number, half_verse_nodes=nodes, half_verse_typ=typ, half_verse_function=function
    )


class TestTypFunctionMarginalVectors:
    def test_has_the_concatenated_phrase_typ_and_phrase_function_dimension(self):
        psalms = [_psalm(number=1, nodes=(100,), typ=(("NP", "VP"),), function=(("Subj", "Pred"),))]
        vector = typ_function_marginal_vectors(psalms)[100]
        assert vector.shape == (_TYP_DIM + _FUNCTION_DIM,)

    def test_matches_the_independent_phrase_marginal_histograms_concatenated(self):
        psalms = [_psalm(number=1, nodes=(100,), typ=(("NP", "VP"),), function=(("Subj", "Pred"),))]
        vector = typ_function_marginal_vectors(psalms)[100]
        expected = np.concatenate(
            [
                unigram_histogram(("NP", "VP"), _TYP_INDEX, _TYP_DIM),
                unigram_histogram(("Subj", "Pred"), _FUNCTION_INDEX, _FUNCTION_DIM),
            ]
        )
        assert np.allclose(vector, expected)


class TestTypFunctionMarginalPsalmVectors:
    def test_broadcasts_the_same_vector_to_every_half_verse(self):
        psalms = [
            _psalm(
                number=1,
                nodes=(100, 101),
                typ=(("NP",), ("VP",)),
                function=(("Subj",), ("Pred",)),
            )
        ]
        vectors = typ_function_marginal_psalm_vectors(psalms)
        assert np.allclose(vectors[100], vectors[101])
        assert vectors[100].shape == (_TYP_DIM + _FUNCTION_DIM,)
