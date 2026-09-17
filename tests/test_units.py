"""The unit contract: one node, its texts by tier, and one encoder call per dataset."""

from __future__ import annotations

import numpy as np
import pytest

from semantic.units import Unit, embed, texts_at

UNITS = [
    Unit(10, {"consonantal": "a", "cantillation": "A"}),
    Unit(20, {"consonantal": "b", "cantillation": "B"}),
    Unit(30, {"consonantal": "c"}),
]


class TestTextsAt:
    def test_returns_the_tier_in_unit_order(self):
        assert texts_at(UNITS, "consonantal") == ["a", "b", "c"]

    def test_refuses_a_unit_that_lacks_the_tier_naming_its_node(self):
        with pytest.raises(ValueError, match="1 units have no cantillation text, first node 30"):
            texts_at(UNITS, "cantillation")


class TestEmbed:
    def test_keys_one_row_per_unit_by_node_from_one_encoder_call(self):
        calls: list[list[str]] = []

        def encode(texts):
            calls.append(list(texts))
            return np.arange(len(texts) * 2, dtype=np.float32).reshape(len(texts), 2)

        vectors = embed(UNITS, "consonantal", encode)

        assert calls == [["a", "b", "c"]]
        assert list(vectors) == [10, 20, 30]
        assert vectors[20].tolist() == [2.0, 3.0]

    def test_refuses_an_encoder_that_returns_the_wrong_number_of_rows(self):
        with pytest.raises(ValueError, match="returned 1 rows for 3 units"):
            embed(UNITS, "consonantal", lambda texts: np.zeros((1, 2)))

    def test_refuses_an_empty_unit_list_before_touching_the_encoder(self):
        with pytest.raises(ValueError, match="no units"):
            embed([], "consonantal", lambda texts: np.zeros((0, 2)))
