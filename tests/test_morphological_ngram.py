from __future__ import annotations

import numpy as np

from morphological.ngram import (
    bigram_histogram,
    pooled_ngram_psalm_vectors,
    reorder,
    sparse_1_2_3gram,
    sparse_pooled_1_2_3gram,
    trigram_histogram,
    unigram_histogram,
)

_VOCAB = ("a", "b", "c")
_INDEX_OF = {v: i for i, v in enumerate(_VOCAB)}
_DIM = len(_VOCAB)


class TestReorder:
    def test_returns_values_unchanged_when_no_order_given(self):
        assert reorder(("a", "b"), 100, None) == ("a", "b")

    def test_applies_the_permutation_for_the_given_node(self):
        assert reorder(("a", "b", "c"), 100, {100: np.array([2, 0, 1])}) == ("c", "a", "b")

    def test_leaves_values_unchanged_when_node_has_no_entry(self):
        assert reorder(("a", "b"), 100, {200: np.array([1, 0])}) == ("a", "b")


class TestUnigramHistogram:
    def test_sums_to_one_and_counts_repeats(self):
        histogram = unigram_histogram(("a", "a", "b"), _INDEX_OF, _DIM)
        assert np.isclose(histogram.sum(), 1.0)
        assert np.isclose(histogram[_INDEX_OF["a"]], 2 / 3)


class TestBigramHistogram:
    def test_degenerates_to_zero_below_two_values(self):
        assert bigram_histogram(("a",), _INDEX_OF, _DIM).sum() == 0.0

    def test_is_order_sensitive(self):
        forward = bigram_histogram(("a", "b", "c"), _INDEX_OF, _DIM)
        backward = bigram_histogram(("c", "b", "a"), _INDEX_OF, _DIM)
        assert not np.allclose(forward, backward)


class TestTrigramHistogram:
    def test_degenerates_to_zero_below_three_values(self):
        assert trigram_histogram(("a", "b"), _INDEX_OF, _DIM).sum() == 0.0


class TestPooledNgramPsalmVectors:
    def test_pools_raw_counts_across_colons_before_normalizing_once(self):
        psalm_columns = [((100, 101), (("a",), ("b", "b", "c")))]
        vectors = pooled_ngram_psalm_vectors(
            psalm_columns, orders=(1,), index_of=_INDEX_OF, dim=_DIM, order_by_node=None
        )
        vector = vectors[100]
        assert np.isclose(vector[_INDEX_OF["a"]], 0.25)
        assert np.isclose(vector[_INDEX_OF["b"]], 0.5)
        assert np.isclose(vector[_INDEX_OF["c"]], 0.25)

    def test_broadcasts_the_identical_vector_within_a_psalm(self):
        psalm_columns = [((100, 101), (("a",), ("b",)))]
        vectors = pooled_ngram_psalm_vectors(
            psalm_columns, orders=(1,), index_of=_INDEX_OF, dim=_DIM, order_by_node=None
        )
        assert np.allclose(vectors[100], vectors[101])


def _dense_from_sparse(indices: np.ndarray, values: np.ndarray, dim: int) -> np.ndarray:
    dense = np.zeros(dim, dtype=np.float32)
    dense[indices] = values
    return dense


def _dense_1_2_3gram(colon: tuple[str, ...]) -> np.ndarray:
    return np.concatenate(
        [
            unigram_histogram(colon, _INDEX_OF, _DIM),
            bigram_histogram(colon, _INDEX_OF, _DIM),
            trigram_histogram(colon, _INDEX_OF, _DIM),
        ]
    )


class TestSparse123Gram:
    def test_matches_the_dense_concatenation_exactly_for_a_typical_colon(self):
        colon = ("a", "b", "c", "a", "b")
        combined_dim = _DIM + _DIM**2 + _DIM**3

        indices, values = sparse_1_2_3gram(colon, _INDEX_OF, _DIM)

        assert np.array_equal(
            _dense_from_sparse(indices, values, combined_dim), _dense_1_2_3gram(colon)
        )

    def test_matches_the_dense_concatenation_for_a_colon_with_repeated_bigrams(self):
        colon = ("a", "b", "a", "b", "a", "b")
        combined_dim = _DIM + _DIM**2 + _DIM**3

        indices, values = sparse_1_2_3gram(colon, _INDEX_OF, _DIM)

        assert np.array_equal(
            _dense_from_sparse(indices, values, combined_dim), _dense_1_2_3gram(colon)
        )

    def test_empty_colon_gives_no_nonzero_entries(self):
        indices, values = sparse_1_2_3gram((), _INDEX_OF, _DIM)
        assert indices.size == 0
        assert values.size == 0

    def test_single_word_colon_gives_only_a_unigram_entry(self):
        indices, values = sparse_1_2_3gram(("a",), _INDEX_OF, _DIM)
        assert indices.tolist() == [_INDEX_OF["a"]]
        assert np.isclose(values[0], 1.0)

    def test_two_word_colon_gives_unigram_and_bigram_but_no_trigram_entries(self):
        combined_dim = _DIM + _DIM**2 + _DIM**3
        indices, values = sparse_1_2_3gram(("a", "b"), _INDEX_OF, _DIM)
        assert np.array_equal(
            _dense_from_sparse(indices, values, combined_dim), _dense_1_2_3gram(("a", "b"))
        )
        assert indices.max() < _DIM + _DIM**2

    def test_never_returns_a_zero_value(self):
        colon = ("a", "b", "c", "a", "c", "b", "b")
        _, values = sparse_1_2_3gram(colon, _INDEX_OF, _DIM)
        assert np.all(values != 0)

    def test_dtypes_are_int32_indices_and_float32_values(self):
        indices, values = sparse_1_2_3gram(("a", "b", "c"), _INDEX_OF, _DIM)
        assert indices.dtype == np.int32
        assert values.dtype == np.float32


class TestSparsePooled123Gram:
    def test_matches_the_dense_pooled_vector_exactly(self):
        combined_dim = _DIM + _DIM**2 + _DIM**3
        psalm_columns = [((100, 101), (("a", "b", "a"), ("b", "c", "a", "b")))]

        sparse_vectors = sparse_pooled_1_2_3gram(psalm_columns, _INDEX_OF, _DIM, order_by_node=None)
        dense_vectors = pooled_ngram_psalm_vectors(
            psalm_columns, orders=(1, 2, 3), index_of=_INDEX_OF, dim=_DIM, order_by_node=None
        )

        sparse_idx, sparse_val = sparse_vectors[100]
        reconstructed = np.zeros(combined_dim, dtype=np.float32)
        reconstructed[sparse_idx] = sparse_val
        assert np.array_equal(reconstructed, dense_vectors[100])

    def test_broadcasts_the_identical_sparse_vector_within_a_psalm(self):
        psalm_columns = [((100, 101), (("a", "b"), ("c",)))]

        vectors = sparse_pooled_1_2_3gram(psalm_columns, _INDEX_OF, _DIM, order_by_node=None)

        assert np.array_equal(vectors[100][0], vectors[101][0])
        assert np.array_equal(vectors[100][1], vectors[101][1])

    def test_applies_order_by_node_per_colon_before_pooling(self):
        psalm_columns = [((100,), (("a", "b", "c"),))]
        order = {100: np.array([2, 1, 0])}

        unshuffled = sparse_pooled_1_2_3gram(psalm_columns, _INDEX_OF, _DIM, order_by_node=None)
        shuffled = sparse_pooled_1_2_3gram(psalm_columns, _INDEX_OF, _DIM, order_by_node=order)

        assert not (
            np.array_equal(unshuffled[100][0], shuffled[100][0])
            and np.array_equal(unshuffled[100][1], shuffled[100][1])
        )
