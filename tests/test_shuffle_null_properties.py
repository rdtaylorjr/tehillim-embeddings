"""The properties the order-shuffle null rests on: content is preserved, only its order moves."""

from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np
import pytest

from core.ngram import bigram_histogram, reorder, unigram_histogram
from core.shuffle import shuffled_order_by_psalm, shuffled_within_half_verse_order
from core.vocabulary import index_map

VOCAB = ("a", "b", "c", "d")
INDEX_OF = index_map(VOCAB)
DIM = len(VOCAB)


@dataclass(frozen=True, slots=True)
class _Psalm:
    number: int
    half_verse_nodes: tuple[int, ...]
    half_verses: tuple[tuple[str, ...], ...]


def _corpus() -> list[_Psalm]:
    rng = np.random.default_rng(7)
    psalms = []
    for number in range(1, 6):
        n_half_verses = int(rng.integers(3, 7))
        half_verses = tuple(
            tuple(rng.choice(VOCAB, size=int(rng.integers(4, 9))).tolist())
            for _ in range(n_half_verses)
        )
        nodes = tuple(number * 100 + i for i in range(n_half_verses))
        psalms.append(_Psalm(number, nodes, half_verses))
    return psalms


class TestWithinHalfVerseShuffle:
    @pytest.mark.parametrize("seed", [1, 2, 17, 999])
    def test_reordering_preserves_every_half_verses_content_exactly(self, seed: int) -> None:
        """A null that changed content would compare content, not order, and so would lie."""
        psalms = _corpus()
        order = shuffled_within_half_verse_order(
            psalms, seed, half_verses=lambda psalm: psalm.half_verses
        )

        for psalm in psalms:
            for node, half_verse in zip(psalm.half_verse_nodes, psalm.half_verses, strict=True):
                shuffled = reorder(half_verse, node, order)

                assert collections.Counter(shuffled) == collections.Counter(half_verse)

    @pytest.mark.parametrize("seed", [1, 2, 17, 999])
    def test_a_unigram_histogram_is_unchanged_by_the_shuffle(self, seed: int) -> None:
        """Order-invariance is why the inventory constructions carry no shuffle control."""
        psalms = _corpus()
        order = shuffled_within_half_verse_order(
            psalms, seed, half_verses=lambda psalm: psalm.half_verses
        )

        for psalm in psalms:
            for node, half_verse in zip(psalm.half_verse_nodes, psalm.half_verses, strict=True):
                np.testing.assert_array_equal(
                    unigram_histogram(reorder(half_verse, node, order), INDEX_OF, DIM),
                    unigram_histogram(half_verse, INDEX_OF, DIM),
                )

    def test_a_bigram_histogram_does_move_so_the_control_is_not_vacuous(self) -> None:
        """If order carried no signal into the bigram block, the null would test nothing."""
        psalms = _corpus()
        moved = 0
        for seed in range(1, 21):
            order = shuffled_within_half_verse_order(
                psalms, seed, half_verses=lambda psalm: psalm.half_verses
            )
            for psalm in psalms:
                for node, half_verse in zip(psalm.half_verse_nodes, psalm.half_verses, strict=True):
                    before = bigram_histogram(half_verse, INDEX_OF, DIM)
                    after = bigram_histogram(reorder(half_verse, node, order), INDEX_OF, DIM)
                    moved += int(not np.array_equal(before, after))

        assert moved > 0

    def test_every_permutation_is_a_bijection_of_the_positions_it_covers(self) -> None:
        psalms = _corpus()
        order = shuffled_within_half_verse_order(
            psalms, 3, half_verses=lambda psalm: psalm.half_verses
        )

        for psalm in psalms:
            for node, half_verse in zip(psalm.half_verse_nodes, psalm.half_verses, strict=True):
                assert sorted(order[node].tolist()) == list(range(len(half_verse)))


class TestHalfVerseOrderShuffle:
    @pytest.mark.parametrize("seed", [1, 2, 17, 999])
    def test_a_psalm_keeps_every_half_verse_it_started_with(self, seed: int) -> None:
        psalms = _corpus()

        order = shuffled_order_by_psalm(psalms, seed)

        for psalm in psalms:
            assert sorted(order[psalm.number].tolist()) == list(range(len(psalm.half_verse_nodes)))

    def test_two_psalms_with_the_same_length_still_permute_independently(self) -> None:
        """Seeding by psalm number keeps one psalm's order from dictating another's."""
        same_length = [
            _Psalm(1, (100, 101, 102, 103, 104), ()),
            _Psalm(2, (200, 201, 202, 203, 204), ()),
        ]

        order = shuffled_order_by_psalm(same_length, 5)

        assert not np.array_equal(order[1], order[2])
