from __future__ import annotations

import numpy as np

from lexical.surface_corpus import SurfacePsalm
from lexical.surface_zoning import surface_position_mean_vectors


def _psalm(*, number, consonantal, nodes):
    return SurfacePsalm(
        number=number,
        half_verse_consonantal=consonantal,
        half_verse_vocalized=consonantal,
        half_verse_cantillation=consonantal,
        half_verse_nodes=nodes,
    )


class TestSurfacePositionMeanVectors:
    def test_dimension_is_twice_the_vocabulary_size(self):
        psalms = [_psalm(number=1, consonantal=(("א",),), nodes=(100,))]
        vocabulary = ("א", "ב", "ג")
        icf_weights = {"א": 1.0, "ב": 1.0, "ג": 1.0}

        vectors = surface_position_mean_vectors(psalms, vocabulary, "consonantal", icf_weights)

        assert len(vectors[100]) == 6

    def test_gives_each_colon_of_a_psalm_its_own_vector(self):
        psalms = [_psalm(number=1, consonantal=(("א", "ב"), ("א",)), nodes=(100, 101))]
        vocabulary = ("א", "ב")
        icf_weights = {"א": 1.5, "ב": 2.0}

        vectors = surface_position_mean_vectors(psalms, vocabulary, "consonantal", icf_weights)

        assert not np.array_equal(vectors[100], vectors[101])
