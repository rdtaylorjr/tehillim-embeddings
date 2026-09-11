from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from core.columns import PsalmColumns
from core.ngram import (
    bigram_histogram,
    concatenated_1_2_3gram_dim,
    ngram_psalm_vectors,
    ngram_vectors,
    pooled_ngram_psalm_vectors,
    reorder,
    sparse_1_2_3gram,
    sparse_ngram_psalm_vectors,
    sparse_ngram_vectors,
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

    def test_rejects_a_permutation_that_does_not_cover_the_sequence(self):
        """A shorter permutation would silently drop values and make the null test nothing."""
        with pytest.raises(ValueError, match="permutation of length 2"):
            reorder(("a", "b", "c"), 100, {100: np.array([1, 0])})


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
    def test_pools_raw_counts_across_half_verses_before_normalizing_once(self):
        psalm_columns = [PsalmColumns(1, (100, 101), (("a",), ("b", "b", "c")))]
        vectors = pooled_ngram_psalm_vectors(psalm_columns, (1,), _VOCAB, order_by_node=None)
        vector = vectors[100]
        assert np.isclose(vector[_INDEX_OF["a"]], 0.25)
        assert np.isclose(vector[_INDEX_OF["b"]], 0.5)
        assert np.isclose(vector[_INDEX_OF["c"]], 0.25)

    def test_broadcasts_the_identical_vector_within_a_psalm(self):
        psalm_columns = [PsalmColumns(1, (100, 101), (("a",), ("b",)))]
        vectors = pooled_ngram_psalm_vectors(psalm_columns, (1,), _VOCAB, order_by_node=None)
        assert np.array_equal(vectors[100], vectors[101])


def _dense_from_sparse(indices: np.ndarray, values: np.ndarray, dim: int) -> np.ndarray:
    dense = np.zeros(dim, dtype=np.float32)
    dense[indices] = values
    return dense


def _dense_1_2_3gram(half_verse: tuple[str, ...]) -> np.ndarray:
    return np.concatenate(
        [
            unigram_histogram(half_verse, _INDEX_OF, _DIM),
            bigram_histogram(half_verse, _INDEX_OF, _DIM),
            trigram_histogram(half_verse, _INDEX_OF, _DIM),
        ]
    )


class TestSparse123Gram:
    def test_matches_the_dense_concatenation_exactly_for_a_typical_half_verse(self):
        half_verse = ("a", "b", "c", "a", "b")
        combined_dim = _DIM + _DIM**2 + _DIM**3

        indices, values = sparse_1_2_3gram(half_verse, _INDEX_OF, _DIM)

        assert np.array_equal(
            _dense_from_sparse(indices, values, combined_dim), _dense_1_2_3gram(half_verse)
        )

    def test_matches_the_dense_concatenation_for_a_half_verse_with_repeated_bigrams(self):
        half_verse = ("a", "b", "a", "b", "a", "b")
        combined_dim = _DIM + _DIM**2 + _DIM**3

        indices, values = sparse_1_2_3gram(half_verse, _INDEX_OF, _DIM)

        assert np.array_equal(
            _dense_from_sparse(indices, values, combined_dim), _dense_1_2_3gram(half_verse)
        )

    def test_empty_half_verse_gives_no_nonzero_entries(self):
        indices, values = sparse_1_2_3gram((), _INDEX_OF, _DIM)
        assert indices.size == 0
        assert values.size == 0

    def test_single_word_half_verse_gives_only_a_unigram_entry(self):
        indices, values = sparse_1_2_3gram(("a",), _INDEX_OF, _DIM)
        assert indices.tolist() == [_INDEX_OF["a"]]
        assert np.isclose(values[0], 1.0)

    def test_two_word_half_verse_gives_unigram_and_bigram_but_no_trigram_entries(self):
        combined_dim = _DIM + _DIM**2 + _DIM**3
        indices, values = sparse_1_2_3gram(("a", "b"), _INDEX_OF, _DIM)
        assert np.array_equal(
            _dense_from_sparse(indices, values, combined_dim), _dense_1_2_3gram(("a", "b"))
        )
        assert indices.max() < _DIM + _DIM**2

    def test_never_returns_a_zero_value(self):
        half_verse = ("a", "b", "c", "a", "c", "b", "b")
        _, values = sparse_1_2_3gram(half_verse, _INDEX_OF, _DIM)
        assert np.all(values != 0)

    def test_dtypes_are_int32_indices_and_float32_values(self):
        indices, values = sparse_1_2_3gram(("a", "b", "c"), _INDEX_OF, _DIM)
        assert indices.dtype == np.int32
        assert values.dtype == np.float32


class TestSparsePooled123Gram:
    def test_matches_the_dense_pooled_vector_exactly(self):
        combined_dim = _DIM + _DIM**2 + _DIM**3
        psalm_columns = [PsalmColumns(1, (100, 101), (("a", "b", "a"), ("b", "c", "a", "b")))]

        sparse_vectors = sparse_pooled_1_2_3gram(psalm_columns, _VOCAB, order_by_node=None)
        dense_vectors = pooled_ngram_psalm_vectors(
            psalm_columns, (1, 2, 3), _VOCAB, order_by_node=None
        )

        sparse_idx, sparse_val = sparse_vectors[100]
        reconstructed = np.zeros(combined_dim, dtype=np.float32)
        reconstructed[sparse_idx] = sparse_val
        assert np.array_equal(reconstructed, dense_vectors[100])

    def test_broadcasts_the_identical_sparse_vector_within_a_psalm(self):
        psalm_columns = [PsalmColumns(1, (100, 101), (("a", "b"), ("c",)))]

        vectors = sparse_pooled_1_2_3gram(psalm_columns, _VOCAB, order_by_node=None)

        assert np.array_equal(vectors[100][0], vectors[101][0])
        assert np.array_equal(vectors[100][1], vectors[101][1])

    def test_applies_order_by_node_per_half_verse_before_pooling(self):
        psalm_columns = [PsalmColumns(1, (100,), (("a", "b", "c"),))]
        order = {100: np.array([2, 1, 0])}

        unshuffled = sparse_pooled_1_2_3gram(psalm_columns, _VOCAB, order_by_node=None)
        shuffled = sparse_pooled_1_2_3gram(psalm_columns, _VOCAB, order_by_node=order)

        assert not (
            np.array_equal(unshuffled[100][0], shuffled[100][0])
            and np.array_equal(unshuffled[100][1], shuffled[100][1])
        )


def test_concatenated_1_2_3gram_dim_is_the_width_the_sparse_layout_actually_fills() -> None:
    """The declared width must cover the trigram block the sparse builder offsets into."""
    dim = 4

    assert concatenated_1_2_3gram_dim(dim) == dim + dim * dim + dim * dim * dim


def test_concatenated_1_2_3gram_dim_leaves_no_index_out_of_range() -> None:
    """Every index a 1+2+3-gram sparse vector can emit must fall inside the declared width."""
    index_of = {"a": 0, "b": 1}
    dim = len(index_of)

    indices, _ = sparse_1_2_3gram(("a", "b", "a", "b"), index_of, dim)

    assert indices.max() < concatenated_1_2_3gram_dim(dim)


@dataclass(frozen=True, slots=True)
class _Psalm:
    number: int
    half_verse_nodes: tuple[int, ...]
    columns: tuple[tuple[str, ...], ...]


def _columns_of(psalm: _Psalm) -> tuple[tuple[str, ...], ...]:
    return psalm.columns


_PSALMS = [_Psalm(1, (10, 11), (("a", "b"), ("b", "c", "a")))]


class TestNgramVectors:
    def test_one_vector_per_node_at_the_concatenated_width(self):
        vectors = ngram_vectors(_PSALMS, _columns_of, _VOCAB, (1, 2))

        assert sorted(vectors) == [10, 11]
        assert all(v.shape == (_DIM + _DIM * _DIM,) for v in vectors.values())

    def test_a_single_order_is_not_concatenated(self):
        vectors = ngram_vectors(_PSALMS, _columns_of, _VOCAB, (1,))

        assert vectors[10].shape == (_DIM,)

    def test_matches_the_histogram_it_concatenates(self):
        vectors = ngram_vectors(_PSALMS, _columns_of, _VOCAB, (1,))

        assert np.allclose(vectors[10], unigram_histogram(("a", "b"), _INDEX_OF, _DIM))

    def test_a_permutation_moves_a_bigram_but_not_a_unigram(self):
        order = {11: np.array([2, 1, 0])}
        unigram = ngram_vectors(_PSALMS, _columns_of, _VOCAB, (1,), order)
        bigram = ngram_vectors(_PSALMS, _columns_of, _VOCAB, (2,), order)

        assert np.allclose(unigram[11], ngram_vectors(_PSALMS, _columns_of, _VOCAB, (1,))[11])
        assert not np.allclose(bigram[11], ngram_vectors(_PSALMS, _columns_of, _VOCAB, (2,))[11])


class TestNgramPsalmVectors:
    def test_broadcasts_one_pooled_vector_across_the_psalms_nodes(self):
        vectors = ngram_psalm_vectors(_PSALMS, _columns_of, _VOCAB, (1,))

        assert np.array_equal(vectors[10], vectors[11])

    def test_equals_pooling_the_same_columns_directly(self):
        columns = [PsalmColumns(1, (10, 11), (("a", "b"), ("b", "c", "a")))]

        vectors = ngram_psalm_vectors(_PSALMS, _columns_of, _VOCAB, (1, 2))

        assert np.allclose(
            vectors[10], pooled_ngram_psalm_vectors(columns, (1, 2), _VOCAB, None)[10]
        )


class TestSparseNgramVectors:
    def test_densifies_to_the_dense_trigram_concatenation(self):
        sparse = sparse_ngram_vectors(_PSALMS, _columns_of, _VOCAB)
        dense = ngram_vectors(_PSALMS, _columns_of, _VOCAB, (1, 2, 3))

        for node, (indices, values) in sparse.items():
            restored = np.zeros(concatenated_1_2_3gram_dim(_DIM), dtype="<f4")
            restored[indices] = values
            assert np.allclose(restored, dense[node])

    def test_psalm_pooled_sparse_matches_its_dense_counterpart(self):
        sparse = sparse_ngram_psalm_vectors(_PSALMS, _columns_of, _VOCAB)
        dense = ngram_psalm_vectors(_PSALMS, _columns_of, _VOCAB, (1, 2, 3))

        for node, (indices, values) in sparse.items():
            restored = np.zeros(concatenated_1_2_3gram_dim(_DIM), dtype="<f4")
            restored[indices] = values
            assert np.allclose(restored, dense[node])
