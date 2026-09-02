from __future__ import annotations

import numpy as np
import pytest

from core.shuffle import (
    DEFAULT_N_SHUFFLES,
    SHUFFLE_NAME_WIDTH,
    shuffle_construction_name,
    shuffled_order_by_psalm,
    shuffled_within_half_verse_order,
)


class _Psalm:
    def __init__(self, number: int, half_verse_nodes: tuple[int, ...]) -> None:
        self.number = number
        self.half_verse_nodes = half_verse_nodes


class TestShuffleConstructionName:
    def test_pads_a_single_digit_seed_to_the_full_width(self):
        assert shuffle_construction_name("icf_position4", 1) == "icf_position4_shuffle0001"

    def test_pads_a_four_digit_seed_without_truncating(self):
        assert shuffle_construction_name("icf_position4", 1000) == "icf_position4_shuffle1000"

    def test_every_seed_up_to_the_default_yields_the_same_name_length(self):
        names = [shuffle_construction_name("x", seed) for seed in range(1, DEFAULT_N_SHUFFLES + 1)]
        assert len({len(name) for name in names}) == 1

    def test_names_sort_lexicographically_in_seed_order(self):
        seeds = list(range(1, DEFAULT_N_SHUFFLES + 1))
        names = [shuffle_construction_name("x", seed) for seed in seeds]
        assert sorted(names) == names

    def test_the_width_admits_every_seed_the_default_produces(self):
        assert DEFAULT_N_SHUFFLES <= 10**SHUFFLE_NAME_WIDTH - 1

    def test_the_name_does_not_depend_on_how_many_shuffles_were_requested(self):
        assert shuffle_construction_name("x", 7) == "x_shuffle0007"

    def test_a_seed_wider_than_the_name_width_is_rejected(self):
        with pytest.raises(ValueError, match="exceeds"):
            shuffle_construction_name("x", 10**SHUFFLE_NAME_WIDTH)

    def test_a_seed_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            shuffle_construction_name("x", 0)


class TestDefaultNShuffles:
    def test_the_permutation_p_value_floor_clears_the_smallest_bh_critical_value(self):
        p_value_floor = 1 / (DEFAULT_N_SHUFFLES + 1)
        smallest_bh_critical_value = 0.05 / 7
        assert p_value_floor <= smallest_bh_critical_value


class TestShuffledOrderByPsalm:
    def test_returns_one_permutation_per_psalm(self):
        psalms = [_Psalm(1, (10, 11, 12)), _Psalm(2, (20, 21))]

        order = shuffled_order_by_psalm(psalms, seed=1)

        assert sorted(order[1].tolist()) == [0, 1, 2]
        assert sorted(order[2].tolist()) == [0, 1]

    def test_is_deterministic_for_a_fixed_seed(self):
        psalms = [_Psalm(1, (10, 11, 12, 13, 14))]

        first = shuffled_order_by_psalm(psalms, seed=7)
        second = shuffled_order_by_psalm(psalms, seed=7)

        assert np.array_equal(first[1], second[1])

    def test_differs_between_seeds(self):
        psalms = [_Psalm(1, tuple(range(20)))]

        first = shuffled_order_by_psalm(psalms, seed=1)
        second = shuffled_order_by_psalm(psalms, seed=2)

        assert not np.array_equal(first[1], second[1])

    def test_a_psalms_permutation_is_independent_of_the_other_psalms_present(self):
        alone = shuffled_order_by_psalm([_Psalm(5, tuple(range(8)))], seed=3)
        together = shuffled_order_by_psalm([_Psalm(1, (0, 1)), _Psalm(5, tuple(range(8)))], seed=3)

        assert np.array_equal(alone[5], together[5])


class _HalfVersePsalm:
    def __init__(
        self, half_verse_nodes: tuple[int, ...], half_verses: tuple[tuple[str, ...], ...]
    ) -> None:
        self.half_verse_nodes = half_verse_nodes
        self.half_verses = half_verses


class TestShuffledWithinHalfVerseOrder:
    def test_returns_one_permutation_per_half_verse_node(self):
        psalms = [_HalfVersePsalm((10, 11), (("a", "b", "c"), ("d", "e")))]

        order = shuffled_within_half_verse_order(
            psalms, seed=1, half_verses=lambda p: p.half_verses
        )

        assert sorted(order[10].tolist()) == [0, 1, 2]
        assert sorted(order[11].tolist()) == [0, 1]

    def test_is_keyed_by_node_so_the_same_node_permutes_alike_across_families(self):
        first = shuffled_within_half_verse_order(
            [_HalfVersePsalm((42,), (tuple("abcdefgh"),))],
            seed=2,
            half_verses=lambda p: p.half_verses,
        )
        second = shuffled_within_half_verse_order(
            [_HalfVersePsalm((42,), (tuple("12345678"),))],
            seed=2,
            half_verses=lambda p: p.half_verses,
        )

        assert np.array_equal(first[42], second[42])

    def test_is_deterministic_for_a_fixed_seed(self):
        psalms = [_HalfVersePsalm((10,), (tuple("abcdef"),))]

        first = shuffled_within_half_verse_order(
            psalms, seed=5, half_verses=lambda p: p.half_verses
        )
        second = shuffled_within_half_verse_order(
            psalms, seed=5, half_verses=lambda p: p.half_verses
        )

        assert np.array_equal(first[10], second[10])
