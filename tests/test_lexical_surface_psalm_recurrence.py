from __future__ import annotations

import numpy as np

from lexical.surface_corpus import SurfacePsalm
from lexical.surface_psalm_recurrence import surface_psalm_spacing_profile_vectors


def _psalm(*, number, consonantal, nodes):
    return SurfacePsalm(
        number=number,
        half_verse_consonantal=consonantal,
        half_verse_vocalized=consonantal,
        half_verse_cantillation=consonantal,
        half_verse_nodes=nodes,
    )


class TestSurfacePsalmSpacingProfileVectors:
    def test_broadcasts_the_same_vector_to_every_colon_of_a_psalm(self):
        psalms = [_psalm(number=1, consonantal=(("א",), ("ב",), ("א",)), nodes=(100, 101, 102))]
        vocabulary = ("א", "ב")
        icf_weights = {"א": 1.0, "ב": 1.0}

        vectors = surface_psalm_spacing_profile_vectors(
            psalms, vocabulary, "consonantal", icf_weights, k=2
        )

        assert np.array_equal(vectors[100], vectors[101])
        assert np.array_equal(vectors[101], vectors[102])

    def test_single_colon_psalm_gives_a_zero_profile(self):
        psalms = [_psalm(number=1, consonantal=(("א",),), nodes=(100,))]
        vocabulary = ("א",)
        icf_weights = {"א": 1.0}

        vectors = surface_psalm_spacing_profile_vectors(
            psalms, vocabulary, "consonantal", icf_weights, k=3
        )

        assert np.array_equal(vectors[100], [0.0, 0.0, 0.0])
