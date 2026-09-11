"""Holds the five domain corpora to one shape, so a reader learns them once."""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
CORPUS_MODULES = [
    "lexical/corpus.py",
    "lexical/surface_corpus.py",
    "morphological/corpus.py",
    "semantic/corpus.py",
    "syntactic/corpus.py",
]


def _psalm_classes() -> list[tuple[str, ast.ClassDef]]:
    found = []
    for relative in CORPUS_MODULES:
        tree = ast.parse((SRC / relative).read_text())
        found += [
            (relative, node)
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name.endswith("Psalm")
        ]
    return found


PSALM_CLASSES = _psalm_classes()
IDS = [node.name for _, node in PSALM_CLASSES]


def _fields(node: ast.ClassDef) -> list[str]:
    return [item.target.id for item in node.body if isinstance(item, ast.AnnAssign)]


@pytest.mark.parametrize(("module", "node"), PSALM_CLASSES, ids=IDS)
def test_every_psalm_type_names_its_domain(module: str, node: ast.ClassDef) -> None:
    """A bare `Psalm` would be ambiguous the moment two domains meet in one import list."""
    assert node.name != "Psalm"


@pytest.mark.parametrize(("module", "node"), PSALM_CLASSES, ids=IDS)
def test_every_psalm_type_leads_with_the_two_fields_they_all_share(
    module: str, node: ast.ClassDef
) -> None:
    """Number and half_verse_nodes identify a psalm, so they lead in every domain alike."""
    assert _fields(node)[:2] == ["number", "half_verse_nodes"]


@pytest.mark.parametrize(("module", "node"), PSALM_CLASSES, ids=IDS)
def test_only_the_psalm_number_is_required(module: str, node: ast.ClassDef) -> None:
    """Every feature tuple defaults to empty, so a test builds only the tiers it exercises."""
    defaults = [item.value is not None for item in node.body if isinstance(item, ast.AnnAssign)]

    assert defaults[0] is False
    assert all(defaults[1:])


@pytest.mark.parametrize(("module", "node"), PSALM_CLASSES, ids=IDS)
def test_every_psalm_type_is_a_frozen_slotted_dataclass(module: str, node: ast.ClassDef) -> None:
    """Frozen so a worker cannot mutate a psalm it was handed, slotted so 150 of them stay small."""
    decorator = next(
        d for d in node.decorator_list if isinstance(d, ast.Call) and d.func.id == "dataclass"
    )
    keywords = {k.arg: k.value.value for k in decorator.keywords}

    assert keywords == {"frozen": True, "slots": True}


def test_every_corpus_module_contributes_a_psalm_type() -> None:
    """A module carries one type per level it reads, so syntactic has both phrase and clause."""
    assert {module for module, _ in PSALM_CLASSES} == set(CORPUS_MODULES)
