from __future__ import annotations

import numpy as np
import pytest

from lexical.surface_corpus import SurfacePsalm
from lexical.surface_vectorize import (
    surface_binary_presence_vectors,
    surface_icf_weighted_vectors,
    surface_log_count_vectors,
    surface_term_frequency_vectors,
    surface_tf_icf_vectors,
)


def _psalm(*, number, consonantal, nodes):
    return SurfacePsalm(
        number=number,
        colon_consonantal=consonantal,
        colon_vocalized=consonantal,
        colon_cantillation=consonantal,
        colon_nodes=nodes,
    )


class TestSurfaceBinaryPresenceVectors:
    def test_marks_presence_of_vocabulary_entries_in_each_colon(self):
        psalms = [_psalm(number=1, consonantal=(("א", "ב"), ("ב",)), nodes=(100, 101))]
        vocabulary = ("א", "ב", "ג")

        vectors = surface_binary_presence_vectors(psalms, vocabulary, "consonantal")

        assert np.array_equal(vectors[100], [1.0, 1.0, 0.0])
        assert np.array_equal(vectors[101], [0.0, 1.0, 0.0])

    def test_repeated_occurrence_within_a_colon_still_reads_as_a_single_one(self):
        psalms = [_psalm(number=1, consonantal=(("א", "א"),), nodes=(100,))]
        vocabulary = ("א",)

        vectors = surface_binary_presence_vectors(psalms, vocabulary, "consonantal")

        assert vectors[100][0] == 1.0

    def test_vector_dimension_matches_vocabulary_size(self):
        psalms = [_psalm(number=1, consonantal=(("א",),), nodes=(100,))]
        vocabulary = ("א", "ב", "ג", "ד")

        vectors = surface_binary_presence_vectors(psalms, vocabulary, "consonantal")

        assert len(vectors[100]) == 4


class TestSurfaceTermFrequencyVectors:
    def test_counts_repeated_occurrences_within_a_colon(self):
        psalms = [_psalm(number=1, consonantal=(("א", "א", "ב"),), nodes=(100,))]
        vocabulary = ("א", "ב")

        vectors = surface_term_frequency_vectors(psalms, vocabulary, "consonantal")

        assert np.array_equal(vectors[100], [2.0, 1.0])

    def test_binary_derives_from_term_frequency(self):
        psalms = [_psalm(number=1, consonantal=(("א", "א"),), nodes=(100,))]
        vocabulary = ("א",)

        binary = surface_binary_presence_vectors(psalms, vocabulary, "consonantal")
        counts = surface_term_frequency_vectors(psalms, vocabulary, "consonantal")

        assert binary[100][0] == 1.0
        assert counts[100][0] == 2.0


class TestSurfaceLogCountVectors:
    def test_applies_log1p_to_term_frequency(self):
        psalms = [_psalm(number=1, consonantal=(("א", "א"),), nodes=(100,))]
        vocabulary = ("א",)

        vectors = surface_log_count_vectors(psalms, vocabulary, "consonantal")

        assert vectors[100][0] == pytest.approx(np.log1p(2.0))


class TestSurfaceIcfWeightedVectors:
    def test_multiplies_binary_presence_by_icf_weight(self):
        psalms = [_psalm(number=1, consonantal=(("א", "ב"),), nodes=(100,))]
        vocabulary = ("א", "ב")
        icf_weights = {"א": 1.5, "ב": 2.0}

        vectors = surface_icf_weighted_vectors(psalms, vocabulary, "consonantal", icf_weights)

        assert np.array_equal(vectors[100], [1.5, 2.0])


class TestSurfaceTfIcfVectors:
    def test_multiplies_log_count_by_icf_weight(self):
        psalms = [_psalm(number=1, consonantal=(("א", "א"),), nodes=(100,))]
        vocabulary = ("א",)
        icf_weights = {"א": 2.0}

        vectors = surface_tf_icf_vectors(psalms, vocabulary, "consonantal", icf_weights)

        assert vectors[100][0] == pytest.approx(np.log1p(2.0) * 2.0)
