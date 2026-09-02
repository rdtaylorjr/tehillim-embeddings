from __future__ import annotations

import numpy as np
import pytest

from core.columns import PsalmColumns
from lexical.constructions import _BUILDERS, FULL_WEIGHTS, vectors_for_weight


def _columns():
    return [PsalmColumns(number=1, nodes=(100, 101), half_verses=(("A", "B"), ("A",)))]


def _icf():
    return {"A": 1.5, "B": 2.0}


class TestDispatchTable:
    def test_every_declared_weight_has_a_builder(self):
        assert set(FULL_WEIGHTS) - set(_BUILDERS) == set()

    def test_no_builder_exists_for_an_undeclared_weight(self):
        assert set(_BUILDERS) - set(FULL_WEIGHTS) == set()

    def test_the_declared_weights_are_unique(self):
        assert len(FULL_WEIGHTS) == len(set(FULL_WEIGHTS))


class TestVectorsForWeight:
    @pytest.mark.parametrize("weight", FULL_WEIGHTS)
    def test_every_weight_builds_a_vector_for_every_half_verse_node(self, weight):
        vectors = vectors_for_weight(_columns(), ("A", "B"), weight, _icf())

        assert set(vectors) == {100, 101}
        assert all(isinstance(v, np.ndarray) for v in vectors.values())

    @pytest.mark.parametrize("weight", FULL_WEIGHTS)
    def test_every_weight_produces_float32(self, weight):
        vectors = vectors_for_weight(_columns(), ("A", "B"), weight, _icf())

        assert all(v.dtype == np.float32 for v in vectors.values())

    def test_an_unknown_weight_is_rejected_by_name(self):
        with pytest.raises(ValueError, match="unknown weight 'nope'"):
            vectors_for_weight(_columns(), ("A", "B"), "nope", _icf())

    def test_the_k_families_differ_by_their_region_count(self):
        two = vectors_for_weight(_columns(), ("A", "B"), "icf_position2", _icf())[100]
        eight = vectors_for_weight(_columns(), ("A", "B"), "icf_position8", _icf())[100]

        assert len(eight) == 4 * len(two)
