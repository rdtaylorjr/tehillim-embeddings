"""One tier vocabulary across the corpus loaders, and a dataset named for the text it holds."""

from __future__ import annotations

import pytest

from core.text import TextTier, strip_accents
from lexical.surface_corpus import SurfacePsalm
from lexical.surface_vocabulary import half_verses_for_tier
from semantic.corpus import SemanticPsalm
from semantic.local_models import select_half_verses
from semantic.registry import VARIATIONS

TIERS: tuple[TextTier, ...] = ("consonantal", "vocalized", "cantillation")

#: One pointed word, its accent-stripped form, and its bare consonants.
CANTILLATED = "בְּרֵאשִׁ֖ית"
POINTED = strip_accents(CANTILLATED)
CONSONANTS = "בראשית"


def test_the_three_tiers_are_genuinely_different_texts() -> None:
    """A tier that collapsed into another would make two datasets identical under two names."""
    assert len({CANTILLATED, POINTED, CONSONANTS}) == 3


def test_every_registry_variation_names_a_known_tier() -> None:
    assert [tier for tier, _ in VARIATIONS] == list(TIERS)


@pytest.mark.parametrize("tier", TIERS)
def test_the_semantic_loader_returns_the_text_its_tier_names(tier: TextTier) -> None:
    psalm = SemanticPsalm(
        number=1,
        half_verses=(CANTILLATED,),
        half_verses_unvocalized=(CONSONANTS,),
        half_verses_niqqud_only=(POINTED,),
        half_verse_nodes=(100,),
    )
    expected = {"consonantal": CONSONANTS, "vocalized": POINTED, "cantillation": CANTILLATED}

    assert select_half_verses(psalm, tier) == (expected[tier],)


@pytest.mark.parametrize("tier", TIERS)
def test_the_surface_loader_returns_the_text_its_tier_names(tier: TextTier) -> None:
    """The two loaders answer to the same tier names, so they must select the same text state."""
    psalm = SurfacePsalm(
        number=1,
        half_verse_consonantal=((CONSONANTS,),),
        half_verse_vocalized=((POINTED,),),
        half_verse_cantillation=((CANTILLATED,),),
        half_verse_nodes=(100,),
    )
    expected = {"consonantal": CONSONANTS, "vocalized": POINTED, "cantillation": CANTILLATED}

    assert half_verses_for_tier(psalm, tier) == ((expected[tier],),)


def test_the_vocalized_tier_is_the_cantillation_tier_with_its_accents_removed() -> None:
    """The two tiers are a controlled comparison, not two unrelated texts."""
    assert strip_accents(CANTILLATED) == POINTED
