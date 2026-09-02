"""The syntax trigram families stored sparsely, asserted to reconstruct the dense form exactly."""

from __future__ import annotations

import numpy as np
import pytest

from core.shuffle import shuffled_within_half_verse_order
from core.support import build_signature_vocabulary
from syntax.corpus import PhrasePsalm
from syntax.function_ngram import (
    phrase_function_1_2_3gram_psalm_sparse_vectors,
    phrase_function_1_2_3gram_psalm_vectors,
    phrase_function_1_2_3gram_sparse_vectors,
    phrase_function_1_2_3gram_vectors,
)
from syntax.signature_vectorize import (
    phrase_signature_1_2_3gram_psalm_sparse_vectors,
    phrase_signature_1_2_3gram_psalm_vectors,
    phrase_signature_1_2_3gram_sparse_vectors,
    phrase_signature_1_2_3gram_vectors,
)
from syntax.typ_ngram import (
    phrase_typ_1_2_3gram_psalm_sparse_vectors,
    phrase_typ_1_2_3gram_psalm_vectors,
    phrase_typ_1_2_3gram_sparse_vectors,
    phrase_typ_1_2_3gram_vectors,
)


def _psalms():
    return [
        PhrasePsalm(
            number=1,
            half_verse_nodes=(100, 101),
            half_verse_typ=(("NP", "VP", "PP"), ("VP", "NP", "CP")),
            half_verse_function=(("Subj", "Pred", "Cmpl"), ("Pred", "Subj", "Conj")),
            half_verse_det=(("det", "NA", "und"), ("NA", "det", "NA")),
            half_verse_rela=(("NA", "NA", "NA"), ("NA", "NA", "NA")),
            half_verse_n_words=((1, 1, 1), (1, 1, 1)),
            half_verse_phrase_id=((10, 11, 12), (13, 14, 15)),
            half_verse_phrase_atom_count=((1, 1, 1), (1, 1, 1)),
            half_verse_subphrase_rela=(("NA",), ("NA",)),
        )
    ]


def _signature_args():
    counts = {"NP|Subj": 5000, "VP|Pred": 5000, "PP|Cmpl": 5000}
    return build_signature_vocabulary(counts, 1000), counts, 1000


def _densify(sparse, dim):
    dense = np.zeros(dim, dtype=np.float32)
    indices, values = sparse
    dense[indices] = values
    return dense


UNIT_PAIRS = [
    ("typ", phrase_typ_1_2_3gram_vectors, phrase_typ_1_2_3gram_sparse_vectors),
    ("function", phrase_function_1_2_3gram_vectors, phrase_function_1_2_3gram_sparse_vectors),
]
PSALM_PAIRS = [
    ("typ", phrase_typ_1_2_3gram_psalm_vectors, phrase_typ_1_2_3gram_psalm_sparse_vectors),
    (
        "function",
        phrase_function_1_2_3gram_psalm_vectors,
        phrase_function_1_2_3gram_psalm_sparse_vectors,
    ),
]


class TestUnitTrigramSparseMatchesDense:
    @pytest.mark.parametrize(("name", "dense_builder", "sparse_builder"), UNIT_PAIRS)
    def test_half_verse_level_reconstructs_the_dense_vector_exactly(
        self, name, dense_builder, sparse_builder
    ):
        psalms = _psalms()
        dense = dense_builder(psalms)
        sparse = sparse_builder(psalms)

        for node in (100, 101):
            assert np.array_equal(_densify(sparse[node], len(dense[node])), dense[node])

    @pytest.mark.parametrize(("name", "dense_builder", "sparse_builder"), PSALM_PAIRS)
    def test_psalm_broadcast_reconstructs_the_dense_vector_exactly(
        self, name, dense_builder, sparse_builder
    ):
        psalms = _psalms()
        dense = dense_builder(psalms)
        sparse = sparse_builder(psalms)

        for node in (100, 101):
            assert np.array_equal(_densify(sparse[node], len(dense[node])), dense[node])

    @pytest.mark.parametrize(("name", "dense_builder", "sparse_builder"), UNIT_PAIRS)
    def test_a_shuffled_order_is_honoured_the_same_way_as_dense(
        self, name, dense_builder, sparse_builder
    ):
        psalms = _psalms()
        order = {100: np.array([2, 1, 0])}

        dense = dense_builder(psalms, order)
        sparse = sparse_builder(psalms, order)

        assert np.array_equal(_densify(sparse[100], len(dense[100])), dense[100])

    @pytest.mark.parametrize(("name", "dense_builder", "sparse_builder"), UNIT_PAIRS)
    def test_the_sparse_form_stores_far_fewer_entries_than_the_dense_dimension(
        self, name, dense_builder, sparse_builder
    ):
        psalms = _psalms()
        dense = dense_builder(psalms)
        sparse = sparse_builder(psalms)

        assert len(sparse[100][0]) < len(dense[100]) / 10


class TestSignatureTrigramSparseMatchesDense:
    def test_half_verse_level_reconstructs_the_dense_vector_exactly(self):
        psalms = _psalms()
        vocabulary, counts, k = _signature_args()

        dense = phrase_signature_1_2_3gram_vectors(psalms, vocabulary, counts, k)
        sparse = phrase_signature_1_2_3gram_sparse_vectors(psalms, vocabulary, counts, k)

        for node in (100, 101):
            assert np.array_equal(_densify(sparse[node], len(dense[node])), dense[node])

    def test_psalm_broadcast_reconstructs_the_dense_vector_exactly(self):
        psalms = _psalms()
        vocabulary, counts, k = _signature_args()

        dense = phrase_signature_1_2_3gram_psalm_vectors(psalms, vocabulary, counts, k)
        sparse = phrase_signature_1_2_3gram_psalm_sparse_vectors(psalms, vocabulary, counts, k)

        for node in (100, 101):
            assert np.array_equal(_densify(sparse[node], len(dense[node])), dense[node])

    def test_a_shuffled_order_is_honoured_the_same_way_as_dense(self):
        psalms = _psalms()
        vocabulary, counts, k = _signature_args()
        order = {100: np.array([2, 1, 0])}

        dense = phrase_signature_1_2_3gram_vectors(psalms, vocabulary, counts, k, order)
        sparse = phrase_signature_1_2_3gram_sparse_vectors(psalms, vocabulary, counts, k, order)

        assert np.array_equal(_densify(sparse[100], len(dense[100])), dense[100])

    def test_psalm_broadcast_gives_every_half_verse_of_a_psalm_the_same_sparse_vector(self):
        psalms = _psalms()
        vocabulary, counts, k = _signature_args()

        sparse = phrase_signature_1_2_3gram_psalm_sparse_vectors(psalms, vocabulary, counts, k)

        assert np.array_equal(sparse[100][0], sparse[101][0])
        assert np.array_equal(sparse[100][1], sparse[101][1])


class TestSparseMatchesDenseUnderAShufflePermutation:
    """The shuffle-null datasets are written from the sparse path, with an order applied."""

    @pytest.mark.parametrize("seed", [1, 2, 7])
    @pytest.mark.parametrize(("name", "dense_builder", "sparse_builder"), UNIT_PAIRS)
    def test_half_verse_level_still_reconstructs_the_dense_vector(
        self, name, dense_builder, sparse_builder, seed
    ):
        psalms = _psalms()
        order = shuffled_within_half_verse_order(
            psalms, seed, half_verses=lambda psalm: psalm.half_verse_typ
        )

        dense = dense_builder(psalms, order)
        sparse = sparse_builder(psalms, order)

        for node in (100, 101):
            assert np.array_equal(_densify(sparse[node], len(dense[node])), dense[node])

    @pytest.mark.parametrize("seed", [1, 2, 7])
    @pytest.mark.parametrize(("name", "dense_builder", "sparse_builder"), PSALM_PAIRS)
    def test_psalm_broadcast_still_reconstructs_the_dense_vector(
        self, name, dense_builder, sparse_builder, seed
    ):
        psalms = _psalms()
        order = shuffled_within_half_verse_order(
            psalms, seed, half_verses=lambda psalm: psalm.half_verse_typ
        )

        dense = dense_builder(psalms, order)
        sparse = sparse_builder(psalms, order)

        for node in (100, 101):
            assert np.array_equal(_densify(sparse[node], len(dense[node])), dense[node])

    @pytest.mark.parametrize("seed", [1, 2, 7])
    def test_the_signature_family_still_reconstructs_the_dense_vector(self, seed):
        psalms = _psalms()
        vocabulary, counts, k = _signature_args()
        order = shuffled_within_half_verse_order(
            psalms, seed, half_verses=lambda psalm: psalm.half_verse_typ
        )

        dense = phrase_signature_1_2_3gram_psalm_vectors(psalms, vocabulary, counts, k, order)
        sparse = phrase_signature_1_2_3gram_psalm_sparse_vectors(
            psalms, vocabulary, counts, k, order
        )

        for node in (100, 101):
            assert np.array_equal(_densify(sparse[node], len(dense[node])), dense[node])
