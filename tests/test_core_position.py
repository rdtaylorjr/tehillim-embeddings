"""Checks normalized half-verse position and binning against their stated definitions."""

from __future__ import annotations

import numpy as np

from core.position import bin_index, half_verse_positions


class TestHalfVersePositions:
    def test_matches_the_continuity_corrected_definition(self) -> None:
        """t_i = (i - 0.5) / n, which keeps the first and last positions off the endpoints."""
        np.testing.assert_allclose(half_verse_positions(4), [0.125, 0.375, 0.625, 0.875])

    def test_no_position_reaches_either_endpoint(self) -> None:
        t = half_verse_positions(10)

        assert t.min() > 0.0
        assert t.max() < 1.0

    def test_positions_are_symmetric_about_the_midpoint(self) -> None:
        t = half_verse_positions(7)

        np.testing.assert_allclose(t + t[::-1], np.ones(7))

    def test_a_single_half_verse_sits_at_the_midpoint(self) -> None:
        np.testing.assert_allclose(half_verse_positions(1), [0.5])

    def test_positions_increase_with_the_half_verse_index(self) -> None:
        assert np.all(np.diff(half_verse_positions(12)) > 0)

    def test_no_half_verses_yields_no_positions(self) -> None:
        assert half_verse_positions(0).size == 0


class TestBinIndex:
    def test_splits_the_unit_interval_into_equal_width_regions(self) -> None:
        t = np.array([0.0, 0.32, 0.34, 0.66, 0.67, 0.99])

        assert list(bin_index(t, 3)) == [0, 0, 1, 1, 2, 2]

    def test_a_position_of_one_lands_in_the_last_bin_rather_than_past_it(self) -> None:
        """Without the clamp this indexes one past the end of a k-column vector."""
        assert bin_index(np.array([1.0]), 3)[0] == 2

    def test_every_position_of_a_real_psalm_lands_inside_the_bins(self) -> None:
        indices = bin_index(half_verse_positions(40), 5)

        assert indices.min() >= 0
        assert indices.max() <= 4

    def test_one_bin_holds_every_position(self) -> None:
        assert set(bin_index(half_verse_positions(9), 1).tolist()) == {0}
