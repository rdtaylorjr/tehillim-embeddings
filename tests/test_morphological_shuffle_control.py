from __future__ import annotations

import numpy as np

from morphological.corpus import MorphologicalPsalm
from morphological.shuffle_control import shuffled_within_colon_order


def _psalm(*, number, words_per_colon, nodes=None):
    nodes = (
        nodes
        if nodes is not None
        else tuple(range(100 * number, 100 * number + len(words_per_colon)))
    )
    return MorphologicalPsalm(
        number=number,
        half_verse_nodes=nodes,
        half_verse_sp=tuple(tuple(f"tag{i}" for i in range(n)) for n in words_per_colon),
    )


class TestShuffledWithinColonOrder:
    def test_returns_a_permutation_of_range_m_for_each_colon(self):
        psalms = [_psalm(number=1, words_per_colon=(5, 3), nodes=(100, 101))]

        order = shuffled_within_colon_order(psalms, seed=1)

        assert sorted(order[100].tolist()) == [0, 1, 2, 3, 4]
        assert sorted(order[101].tolist()) == [0, 1, 2]

    def test_is_deterministic_for_a_fixed_seed(self):
        psalms = [_psalm(number=1, words_per_colon=(6,), nodes=(200,))]

        first = shuffled_within_colon_order(psalms, seed=7)
        second = shuffled_within_colon_order(psalms, seed=7)

        assert np.array_equal(first[200], second[200])

    def test_different_seeds_give_different_orders(self):
        psalms = [_psalm(number=1, words_per_colon=(8,), nodes=(300,))]

        order_a = shuffled_within_colon_order(psalms, seed=1)
        order_b = shuffled_within_colon_order(psalms, seed=2)

        assert not np.array_equal(order_a[300], order_b[300])

    def test_different_colons_get_independent_orders_for_the_same_seed(self):
        psalms = [_psalm(number=1, words_per_colon=(8, 8), nodes=(400, 401))]

        order = shuffled_within_colon_order(psalms, seed=1)

        assert not np.array_equal(order[400], order[401])

    def test_a_single_word_colon_gets_the_trivial_order(self):
        psalms = [_psalm(number=1, words_per_colon=(1,), nodes=(500,))]

        order = shuffled_within_colon_order(psalms, seed=1)

        assert order[500].tolist() == [0]

    def test_preserves_the_multiset_of_pos_tags_when_applied(self):
        psalms = [_psalm(number=1, words_per_colon=(4,), nodes=(600,))]
        colon_sp = psalms[0].half_verse_sp[0]

        order = shuffled_within_colon_order(psalms, seed=3)
        reordered = tuple(colon_sp[i] for i in order[600])

        assert sorted(reordered) == sorted(colon_sp)
