"""The det, rela and subphrase histograms differ only in vocabulary and in what they mask."""

from __future__ import annotations

import numpy as np
import pytest

from syntactic.corpus import PhrasePsalm
from syntactic.det_vectorize import phrase_det_1gram_psalm_vectors, phrase_det_1gram_vectors
from syntactic.rela import SAFE_RELA_VOCABULARY
from syntactic.rela_vectorize import phrase_rela_1gram_psalm_vectors, phrase_rela_1gram_vectors
from syntactic.subphrase import SAFE_SUBPHRASE_RELA_VOCABULARY
from syntactic.subphrase_vectorize import (
    subphrase_rela_1gram_psalm_vectors,
    subphrase_rela_1gram_vectors,
)
from syntactic.vocabulary import DET_VOCABULARY

NODES = (100, 101)


def _psalm(**columns: tuple[tuple[str, ...], ...]) -> PhrasePsalm:
    return PhrasePsalm(number=1, half_verse_nodes=NODES, **columns)


FEATURES = [
    pytest.param(
        _psalm(half_verse_det=(("det", "und"), ("det", "NA", "und"))),
        phrase_det_1gram_vectors,
        phrase_det_1gram_psalm_vectors,
        DET_VOCABULARY,
        id="det",
    ),
    pytest.param(
        _psalm(half_verse_rela=(("Appo", "Para"), ("Link", "NA", "Spec"))),
        phrase_rela_1gram_vectors,
        phrase_rela_1gram_psalm_vectors,
        SAFE_RELA_VOCABULARY,
        id="rela",
    ),
    pytest.param(
        _psalm(half_verse_subphrase_rela=(("adj", "par"), ("atr", "NA", "rec"))),
        subphrase_rela_1gram_vectors,
        subphrase_rela_1gram_psalm_vectors,
        SAFE_SUBPHRASE_RELA_VOCABULARY,
        id="subphrase_rela",
    ),
]


@pytest.mark.parametrize(("psalm", "per_node", "pooled", "vocabulary"), FEATURES)
def test_one_normalized_histogram_per_half_verse_node(psalm, per_node, pooled, vocabulary):
    vectors = per_node([psalm])

    assert sorted(vectors) == list(NODES)
    assert all(v.shape == (len(vocabulary),) for v in vectors.values())
    assert all(np.isclose(v.sum(), 1.0) for v in vectors.values())


@pytest.mark.parametrize(("psalm", "per_node", "pooled", "vocabulary"), FEATURES)
def test_the_pooled_vector_is_broadcast_to_every_node(psalm, per_node, pooled, vocabulary):
    vectors = pooled([psalm])

    assert np.array_equal(vectors[NODES[0]], vectors[NODES[1]])
    assert vectors[NODES[0]].shape == (len(vocabulary),)


@pytest.mark.parametrize(("psalm", "per_node", "pooled", "vocabulary"), FEATURES)
def test_a_masked_value_never_takes_a_slot_of_its_own(psalm, per_node, pooled, vocabulary):
    """`Para` and `par` are counted as NA, so no vector may place weight outside the vocabulary."""
    vectors = per_node([psalm])

    assert all(np.isclose(v.sum(), 1.0) for v in vectors.values())
    assert all(len(v) == len(vocabulary) for v in vectors.values())
