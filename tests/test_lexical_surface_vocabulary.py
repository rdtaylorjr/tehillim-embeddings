from __future__ import annotations

from lexical.surface_corpus import SurfacePsalm
from lexical.surface_vocabulary import build_surface_vocabulary


def _psalm(*, number, consonantal, vocalized=None, cantillation=None):
    return SurfacePsalm(
        number=number,
        colon_consonantal=consonantal,
        colon_vocalized=vocalized or consonantal,
        colon_cantillation=cantillation or consonantal,
        colon_nodes=tuple(range(len(consonantal))),
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

    def test_repeated_forms_within_and_across_cola_count_once(self):
        psalms = [
            _psalm(number=1, consonantal=(("בר", "בר"), ("בר",))),
        ]

        assert build_surface_vocabulary(psalms, tier="consonantal") == ("בר",)

    def test_empty_psalm_list_returns_empty_vocabulary(self):
        assert build_surface_vocabulary([], tier="consonantal") == ()
