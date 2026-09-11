"""Each feature binds one vocabulary and one column, which is all that separates the n-gram sets."""

from __future__ import annotations

import numpy as np
import pytest

from core.ngram import ngram_psalm_vectors, ngram_vectors
from morphological.corpus import MorphologicalPsalm
from morphological.vocabulary import SP_VOCABULARY, sp_columns
from syntactic.corpus import PhrasePsalm
from syntactic.vocabulary import FUNCTION_VOCABULARY, TYP_VOCABULARY, function_columns, typ_columns

NODES = (100, 101)
SP = (("subs", "verb"), ("verb", "subs", "conj"))
TYP = (("NP", "VP"), ("VP", "NP", "CP"))
FUNCTION = (("Subj", "Pred"), ("Pred", "Subj", "Conj"))

FEATURES = [
    pytest.param(
        MorphologicalPsalm(number=1, half_verse_nodes=NODES, half_verse_sp=SP),
        sp_columns,
        SP_VOCABULARY,
        SP,
        id="sp",
    ),
    pytest.param(
        PhrasePsalm(
            number=1, half_verse_nodes=NODES, half_verse_typ=TYP, half_verse_function=FUNCTION
        ),
        typ_columns,
        TYP_VOCABULARY,
        TYP,
        id="typ",
    ),
    pytest.param(
        PhrasePsalm(
            number=1, half_verse_nodes=NODES, half_verse_typ=TYP, half_verse_function=FUNCTION
        ),
        function_columns,
        FUNCTION_VOCABULARY,
        FUNCTION,
        id="function",
    ),
]


@pytest.mark.parametrize(("psalm", "columns_of", "vocabulary", "expected"), FEATURES)
def test_the_selector_reads_its_own_feature_column(psalm, columns_of, vocabulary, expected):
    assert columns_of(psalm) == expected


@pytest.mark.parametrize(("psalm", "columns_of", "vocabulary", "expected"), FEATURES)
def test_vectors_carry_the_width_of_that_features_vocabulary(
    psalm, columns_of, vocabulary, expected
):
    vectors = ngram_vectors([psalm], columns_of, vocabulary, (1,))

    assert sorted(vectors) == list(NODES)
    assert all(v.shape == (len(vocabulary),) for v in vectors.values())


@pytest.mark.parametrize(("psalm", "columns_of", "vocabulary", "expected"), FEATURES)
def test_a_psalm_pooled_vector_is_broadcast_across_the_psalms_nodes(
    psalm, columns_of, vocabulary, expected
):
    vectors = ngram_psalm_vectors([psalm], columns_of, vocabulary, (1, 2))

    assert np.array_equal(vectors[NODES[0]], vectors[NODES[1]])
