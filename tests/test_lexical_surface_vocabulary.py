from __future__ import annotations

from lexical.surface_corpus import SurfacePsalm
from lexical.surface_vocabulary import build_surface_vocabulary, columns_for_tier


def _psalm(*, number, consonantal, vocalized=None, cantillation=None):
    return SurfacePsalm(
        number=number,
        half_verse_consonantal=consonantal,
        half_verse_vocalized=vocalized or consonantal,
        half_verse_cantillation=cantillation or consonantal,
        half_verse_nodes=tuple(range(len(consonantal))),
    )


class TestBuildSurfaceVocabulary:
    def test_returns_sorted_distinct_consonantal_forms_across_all_psalms(self):
        psalms = [
            _psalm(number=1, consonantal=(("בר", "אש"),)),
            _psalm(number=2, consonantal=(("אר", "בר"),)),
        ]

        vocabulary = build_surface_vocabulary(psalms, tier="consonantal")

        assert vocabulary == ("אר", "אש", "בר")

    def test_selects_the_vocalized_tier_independently_of_consonantal(self):
        psalms = [
            _psalm(
                number=1,
                consonantal=(("בר",),),
                vocalized=(("בָר",),),
            ),
        ]

        assert build_surface_vocabulary(psalms, tier="consonantal") == ("בר",)
        assert build_surface_vocabulary(psalms, tier="vocalized") == ("בָר",)

    def test_repeated_forms_within_and_across_half_verses_count_once(self):
        psalms = [
            _psalm(number=1, consonantal=(("בר", "בר"), ("בר",))),
        ]

        assert build_surface_vocabulary(psalms, tier="consonantal") == ("בר",)

    def test_empty_psalm_list_returns_empty_vocabulary(self):
        assert build_surface_vocabulary([], tier="consonantal") == ()


_ALL_TIERS = ("consonantal", "vocalized", "cantillation")


class TestColumnsForTier:
    def test_projects_each_psalm_onto_number_nodes_and_that_tier_s_half_verses(self):
        psalms = [
            SurfacePsalm(
                number=7,
                half_verse_consonantal=(("ab", "cd"),),
                half_verse_vocalized=(("aXb", "cXd"),),
                half_verse_cantillation=(("aYb", "cYd"),),
                half_verse_nodes=(100,),
            )
        ]

        columns = columns_for_tier(psalms, "vocalized")

        assert len(columns) == 1
        assert columns[0].number == 7
        assert columns[0].nodes == (100,)
        assert columns[0].half_verses == (("aXb", "cXd"),)

    def test_each_tier_selects_its_own_word_forms(self):
        psalms = [
            SurfacePsalm(
                number=1,
                half_verse_consonantal=(("ab",),),
                half_verse_vocalized=(("aXb",),),
                half_verse_cantillation=(("aYb",),),
                half_verse_nodes=(100,),
            )
        ]

        selected = {tier: columns_for_tier(psalms, tier)[0].half_verses for tier in _ALL_TIERS}

        assert selected["consonantal"] == (("ab",),)
        assert selected["vocalized"] == (("aXb",),)
        assert selected["cantillation"] == (("aYb",),)

    def test_preserves_psalm_order(self):
        psalms = [
            SurfacePsalm(number=n, half_verse_consonantal=(("a",),), half_verse_nodes=(n,))
            for n in (3, 1, 2)
        ]

        assert [c.number for c in columns_for_tier(psalms, "consonantal")] == [3, 1, 2]
