from __future__ import annotations

import numpy as np

from lexical.corpus import LexicalPsalm
from lexical.shuffle_control import shuffled_order_by_psalm


def _psalm(*, number, n_cola):
    return LexicalPsalm(
        number=number,
        half_verse_lexemes=tuple((f"L{i}",) for i in range(n_cola)),
        half_verse_forms=tuple((f"F{i}",) for i in range(n_cola)),
        half_verse_nodes=tuple(range(100 * number, 100 * number + n_cola)),
    )


class TestShuffledOrderByPsalm:
    def test_returns_a_permutation_of_range_n_for_each_psalm(self):
        psalms = [_psalm(number=1, n_cola=5), _psalm(number=2, n_cola=3)]

        order = shuffled_order_by_psalm(psalms, seed=1)

        assert sorted(order[1].tolist()) == [0, 1, 2, 3, 4]
        assert sorted(order[2].tolist()) == [0, 1, 2]

    def test_is_deterministic_for_a_fixed_seed(self):
        psalms = [_psalm(number=1, n_cola=6)]

        first = shuffled_order_by_psalm(psalms, seed=7)
        second = shuffled_order_by_psalm(psalms, seed=7)

        assert np.array_equal(first[1], second[1])

    def test_different_seeds_give_different_orders(self):
        psalms = [_psalm(number=1, n_cola=8)]

        order_a = shuffled_order_by_psalm(psalms, seed=1)
        order_b = shuffled_order_by_psalm(psalms, seed=2)

        assert not np.array_equal(order_a[1], order_b[1])

    def test_different_psalms_get_independent_orders_for_the_same_seed(self):
        psalms = [_psalm(number=1, n_cola=8), _psalm(number=2, n_cola=8)]

        order = shuffled_order_by_psalm(psalms, seed=1)

        assert not np.array_equal(order[1], order[2])

    def test_a_single_colon_psalm_gets_the_trivial_order(self):
        psalms = [_psalm(number=1, n_cola=1)]

        order = shuffled_order_by_psalm(psalms, seed=1)

        assert order[1].tolist() == [0]
