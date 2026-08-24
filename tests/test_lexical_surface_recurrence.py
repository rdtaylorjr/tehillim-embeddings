from __future__ import annotations

import numpy as np

from lexical.surface_corpus import SurfacePsalm
from lexical.surface_recurrence import surface_spacing_profile_vectors


def _psalm(*, number, consonantal, nodes):
    return SurfacePsalm(
        number=number,
        colon_consonantal=consonantal,
        colon_vocalized=consonantal,
        colon_cantillation=consonantal,
        colon_nodes=nodes,
    )


class TestSurfaceSpacingProfileVectors:
    def test_dimension_equals_k(self):
        psalms = [_psalm(number=1, consonantal=(("א",), ("ב",), ("א",)), nodes=(100, 101, 102))]
        vocabulary = ("א", "ב")
        icf_weights = {"א": 1.0, "ב": 1.0}

        vectors = surface_spacing_profile_vectors(
            psalms, vocabulary, "consonantal", icf_weights, k=2
        )

        assert len(vectors[100]) == 2

    def test_single_colon_psalm_gives_a_zero_profile(self):
        psalms = [_psalm(number=1, consonantal=(("א",),), nodes=(100,))]
        vocabulary = ("א",)
        icf_weights = {"א": 1.0}

        vectors = surface_spacing_profile_vectors(
            psalms, vocabulary, "consonantal", icf_weights, k=3
        )

        assert np.array_equal(vectors[100], [0.0, 0.0, 0.0])
