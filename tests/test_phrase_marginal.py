from __future__ import annotations

import numpy as np

from phrase.corpus import PhrasePsalm
from phrase.function_ngram import phrase_function_unigram_histogram
from phrase.marginal import typ_function_marginal_psalm_vectors, typ_function_marginal_vectors
from phrase.typ_ngram import phrase_typ_unigram_histogram
from phrase.vocabulary import FUNCTION_VOCABULARY, TYP_VOCABULARY

_TYP_DIM = len(TYP_VOCABULARY)
_FUNCTION_DIM = len(FUNCTION_VOCABULARY)


def _psalm(*, number, nodes, typ, function):
    return PhrasePsalm(
        number=number, half_verse_nodes=nodes, half_verse_typ=typ, half_verse_function=function
    )


class TestTypFunctionMarginalVectors:
    def test_has_the_concatenated_typ_and_function_dimension(self):
        psalms = [_psalm(number=1, nodes=(100,), typ=(("NP", "VP"),), function=(("Subj", "Pred"),))]
        vector = typ_function_marginal_vectors(psalms)[100]
        assert vector.shape == (_TYP_DIM + _FUNCTION_DIM,)

    def test_matches_the_independent_marginal_histograms_concatenated(self):
        psalms = [_psalm(number=1, nodes=(100,), typ=(("NP", "VP"),), function=(("Subj", "Pred"),))]
        vector = typ_function_marginal_vectors(psalms)[100]
        expected = np.concatenate(
            [
                phrase_typ_unigram_histogram(("NP", "VP")),
                phrase_function_unigram_histogram(("Subj", "Pred")),
            ]
        )
        assert np.allclose(vector, expected)


class TestTypFunctionMarginalPsalmVectors:
    def test_broadcasts_the_same_vector_to_every_colon(self):
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
