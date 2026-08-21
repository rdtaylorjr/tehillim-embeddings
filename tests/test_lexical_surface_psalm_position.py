from __future__ import annotations

import numpy as np

from lexical.surface_corpus import SurfacePsalm
from lexical.surface_psalm_position import surface_psalm_positional_icf_vectors


def _psalm(*, number, consonantal, nodes):
    return SurfacePsalm(
        number=number,
        half_verse_consonantal=consonantal,
        half_verse_vocalized=consonantal,
        half_verse_cantillation=consonantal,
        half_verse_nodes=nodes,
    )


class TestSurfacePsalmPositionalIcfVectors:
    def test_broadcasts_the_same_vector_to_every_colon_of_a_psalm(self):
        psalms = [_psalm(number=1, consonantal=(("א",), ("ב",)), nodes=(100, 101))]
        vocabulary = ("א", "ב")
        icf_weights = {"א": 1.5, "ב": 2.0}

        vectors = surface_psalm_positional_icf_vectors(
            psalms, vocabulary, "consonantal", icf_weights, k=2
        )

        assert np.array_equal(vectors[100], vectors[101])

    def test_dimension_is_k_times_vocabulary_size(self):
        psalms = [_psalm(number=1, consonantal=(("א",),), nodes=(100,))]
        vocabulary = ("א", "ב")
        icf_weights = {"א": 1.0, "ב": 1.0}

        vectors = surface_psalm_positional_icf_vectors(
            psalms, vocabulary, "consonantal", icf_weights, k=4
        )

        assert len(vectors[100]) == 8
