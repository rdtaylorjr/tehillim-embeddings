"""Checks the vocabulary-to-column indexing that defines every histogram's column order."""

from __future__ import annotations

from core.vocabulary import index_map


def test_columns_follow_the_vocabulary_order_it_was_given() -> None:
    """Column order is the dataset's schema, so it follows the vocabulary rather than sorting."""
    assert index_map(("b", "a", "c")) == {"b": 0, "a": 1, "c": 2}


def test_every_vocabulary_value_gets_a_column() -> None:
    vocabulary = ("a", "b", "c", "d")

    assert sorted(index_map(vocabulary).values()) == list(range(len(vocabulary)))


def test_an_empty_vocabulary_has_no_columns() -> None:
    assert index_map(()) == {}


def test_accepts_a_list_as_readily_as_a_tuple() -> None:
    assert index_map(["a", "b"]) == index_map(("a", "b"))
