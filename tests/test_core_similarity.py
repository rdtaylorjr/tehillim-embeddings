"""Checks the shared cosine and lag geometry against their definitions, not against themselves."""

from __future__ import annotations

import numpy as np

from core.similarity import (
    full_cosine_similarity_matrix,
    half_verse_vector,
    lag_bin_index,
    normalized_lag,
    pairwise_cosine_similarity,
)


def _reference_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """cos(a, b) written straight from the definition, with no shared code to agree with."""
    denominator = float(np.sqrt(a @ a) * np.sqrt(b @ b))
    return 0.0 if denominator == 0 else float(a @ b) / denominator


class TestFullCosineSimilarityMatrix:
    def test_every_entry_equals_the_definition(self) -> None:
        rng = np.random.default_rng(0)
        vectors = rng.normal(size=(5, 4))

        matrix = full_cosine_similarity_matrix(vectors)

        expected = np.array(
            [[_reference_cosine(a, b) for b in vectors] for a in vectors], dtype=float
        )
        np.testing.assert_allclose(matrix, expected, rtol=0, atol=1e-12)

    def test_a_vector_is_exactly_similar_to_itself(self) -> None:
        """The diagonal pins the normalization: a scale error moves it off 1 immediately."""
        rng = np.random.default_rng(1)
        vectors = rng.normal(size=(6, 3))

        matrix = full_cosine_similarity_matrix(vectors)

        np.testing.assert_allclose(np.diag(matrix), np.ones(6), rtol=0, atol=1e-12)

    def test_no_similarity_can_exceed_one(self) -> None:
        rng = np.random.default_rng(2)

        matrix = full_cosine_similarity_matrix(rng.normal(size=(8, 5)))

        assert matrix.max() <= 1.0 + 1e-12

    def test_a_zero_vector_is_similar_to_nothing_rather_than_undefined(self) -> None:
        vectors = np.array([[0.0, 0.0], [1.0, 0.0]])

        matrix = full_cosine_similarity_matrix(vectors)

        assert matrix[0, 0] == 0.0
        assert matrix[0, 1] == 0.0

    def test_the_matrix_is_symmetric(self) -> None:
        rng = np.random.default_rng(3)

        matrix = full_cosine_similarity_matrix(rng.normal(size=(7, 4)))

        np.testing.assert_allclose(matrix, matrix.T, rtol=0, atol=1e-12)

    def test_scaling_a_vector_leaves_its_similarities_unchanged(self) -> None:
        """Cosine reads direction only, so a magnitude change must not move any entry."""
        vectors = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 0.5]])
        scaled = vectors * np.array([[10.0], [1.0], [0.01]])

        np.testing.assert_allclose(
            full_cosine_similarity_matrix(vectors),
            full_cosine_similarity_matrix(scaled),
            rtol=0,
            atol=1e-12,
        )


class TestPairwiseCosineSimilarity:
    def test_returns_the_upper_triangle_of_the_full_matrix_in_triu_order(self) -> None:
        rng = np.random.default_rng(4)
        vectors = rng.normal(size=(5, 3))
        rows, cols = np.triu_indices(5, k=1)

        pairs = pairwise_cosine_similarity(vectors)

        np.testing.assert_allclose(
            pairs, full_cosine_similarity_matrix(vectors)[rows, cols], rtol=0, atol=1e-12
        )

    def test_has_one_entry_per_unordered_pair(self) -> None:
        assert len(pairwise_cosine_similarity(np.eye(6))) == 6 * 5 // 2


class TestNormalizedLag:
    def test_adjacent_half_verses_are_the_smallest_nonzero_separation(self) -> None:
        assert normalized_lag(5).min() == 1 / 4

    def test_the_first_and_last_half_verse_are_a_full_unit_apart(self) -> None:
        """The endpoints define the scale, so |i-j| is divided by n-1 rather than n."""
        assert normalized_lag(5).max() == 1.0

    def test_one_half_verse_has_no_pairs_at_all(self) -> None:
        assert normalized_lag(1).size == 0


class TestLagBinIndex:
    def test_a_separation_of_one_falls_in_the_last_bin_rather_than_past_it(self) -> None:
        assert lag_bin_index(np.array([1.0]), 4)[0] == 3

    def test_zero_separation_falls_in_the_first_bin(self) -> None:
        assert lag_bin_index(np.array([0.0]), 4)[0] == 0

    def test_bins_are_equal_width_across_the_unit_interval(self) -> None:
        deltas = np.array([0.0, 0.24, 0.25, 0.49, 0.5, 0.74, 0.75, 1.0])

        assert list(lag_bin_index(deltas, 4)) == [0, 0, 1, 1, 2, 2, 3, 3]


class TestHalfVerseVector:
    def test_carries_the_weight_of_every_vocabulary_value_present(self) -> None:
        weights = np.array([0.5, 1.5, 2.5])

        vector = half_verse_vector(("a", "c"), {"a": 0, "b": 1, "c": 2}, weights, 3)

        np.testing.assert_array_equal(vector, np.array([0.5, 0.0, 2.5], dtype=np.float32))

    def test_records_presence_rather_than_count(self) -> None:
        """A repeated value marks the same column once, which is what presence weighting means."""
        weights = np.array([0.5, 1.5])

        repeated = half_verse_vector(("a", "a", "a"), {"a": 0, "b": 1}, weights, 2)

        np.testing.assert_array_equal(
            repeated, half_verse_vector(("a",), {"a": 0, "b": 1}, weights, 2)
        )

    def test_ignores_a_value_outside_the_vocabulary(self) -> None:
        weights = np.array([0.5, 1.5])

        vector = half_verse_vector(("a", "zzz"), {"a": 0, "b": 1}, weights, 2)

        np.testing.assert_array_equal(vector, np.array([0.5, 0.0], dtype=np.float32))

    def test_an_empty_half_verse_is_the_zero_vector(self) -> None:
        assert not half_verse_vector((), {"a": 0}, np.array([1.0]), 1).any()
