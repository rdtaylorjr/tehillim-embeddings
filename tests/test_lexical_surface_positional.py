from __future__ import annotations

import numpy as np

from lexical.surface_corpus import SurfacePsalm
from lexical.surface_positional import surface_positional_icf_vectors


def _psalm(*, number, consonantal, nodes):
    return SurfacePsalm(
        number=number,
        half_verse_consonantal=consonantal,
        half_verse_vocalized=consonantal,
        half_verse_cantillation=consonantal,
        half_verse_nodes=nodes,
    )


class TestSurfacePositionalIcfVectors:
    def test_each_colon_nonzero_only_in_its_own_position_bin(self):
        psalms = [_psalm(number=1, consonantal=(("א",), ("ב",)), nodes=(100, 101))]
        vocabulary = ("א", "ב")
        icf_weights = {"א": 1.5, "ב": 2.0}

        vectors = surface_positional_icf_vectors(
            psalms, vocabulary, "consonantal", icf_weights, k=2
        )

        # k=2, dim=2: first colon (t=0.25) lands in bin 0, second (t=0.75) in bin 1.
        assert np.allclose(vectors[100], [1.5, 0.0, 0.0, 0.0])
        assert np.allclose(vectors[101], [0.0, 0.0, 0.0, 2.0])

    def test_dimension_is_k_times_vocabulary_size(self):
        psalms = [_psalm(number=1, consonantal=(("א",),), nodes=(100,))]
        vocabulary = ("א", "ב", "ג")

        icf_weights = {"א": 1.0, "ב": 1.0, "ג": 1.0}
        vectors = surface_positional_icf_vectors(
            psalms, vocabulary, "consonantal", icf_weights, k=4
        )

        assert len(vectors[100]) == 12
