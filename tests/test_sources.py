"""The sources the semantic domain embeds: one per scope, named for their cells."""

from __future__ import annotations

import pytest

from core.partition import BHSA_HALF_VERSE, Scope
from semantic.sources import SOURCES, Source, scope_arguments, source_for

SCROLL = Scope("dss", "verse", witness="11Q5", reconstruction="none")


def test_the_bhsa_source_renders_three_tiers_and_a_scroll_only_consonants() -> None:
    assert source_for(BHSA_HALF_VERSE).tiers == ("consonantal", "vocalized", "cantillation")
    assert source_for(SCROLL).tiers == ("consonantal",)


def test_every_source_has_its_own_scope() -> None:
    assert len({source.scope for source in SOURCES}) == len(SOURCES)


def test_the_bhsa_is_read_at_the_half_verse_and_at_the_verse() -> None:
    assert source_for(Scope("bhsa", "verse")).tiers == source_for(BHSA_HALF_VERSE).tiers
    assert source_for(Scope("bhsa", "verse")).slug == "bhsa-verse"


def test_the_slug_joins_the_scope_values_in_path_order() -> None:
    assert source_for(BHSA_HALF_VERSE).slug == "bhsa-half_verse"
    assert source_for(SCROLL).slug == "dss-11Q5-none-verse"


def test_the_arguments_select_the_scope_key_by_key() -> None:
    assert scope_arguments(SCROLL) == (
        "--corpus=dss",
        "--witness=11Q5",
        "--reconstruction=none",
        "--unit=verse",
    )
    assert scope_arguments(BHSA_HALF_VERSE) == ("--corpus=bhsa", "--unit=half_verse")


def test_an_undeclared_scope_is_refused() -> None:
    with pytest.raises(KeyError, match="no semantic source"):
        source_for(Scope("bhsa", "chapter"))


def test_a_source_loads_its_units_only_when_asked() -> None:
    calls: list[int] = []
    source = Source(SCROLL, ("consonantal",), lambda: calls.append(1) or [])
    assert calls == []
    source.load()
    assert calls == [1]
