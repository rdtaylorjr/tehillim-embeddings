from __future__ import annotations

import pytest

from core.columns import PsalmColumns


class TestPsalmColumns:
    def test_carries_number_nodes_and_half_verses(self):
        columns = PsalmColumns(number=3, nodes=(10, 11), half_verses=(("a",), ("b", "c")))

        assert columns.number == 3
        assert columns.nodes == (10, 11)
        assert columns.half_verses == (("a",), ("b", "c"))

    def test_is_frozen_so_a_vectorizer_cannot_mutate_its_input(self):
        columns = PsalmColumns(number=1, nodes=(10,), half_verses=(("a",),))

        with pytest.raises(AttributeError):
            columns.number = 2

    def test_nodes_and_half_verses_line_up_one_to_one(self):
        columns = PsalmColumns(number=1, nodes=(10, 11, 12), half_verses=(("a",), ("b",), ("c",)))

        assert len(columns.nodes) == len(columns.half_verses)
