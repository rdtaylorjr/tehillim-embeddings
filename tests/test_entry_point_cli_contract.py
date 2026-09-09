"""Holds every entry point to one CLI contract: argv in, corpus injected, shared arguments."""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"

#: Shared once in core.cli, so an entry point restating one is drift rather than configuration.
SHARED_OPTIONS = ("--output-root", "--config-root", "--n-shuffles", "--max-workers")


def _entry_point_modules() -> list[str]:
    """Every module defining a main(), which is what makes a module an entry point."""
    names = []
    for path in sorted(SRC.rglob("*.py")):
        if "egg-info" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        if any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body):
            names.append(str(path.relative_to(SRC).with_suffix("")).replace("/", "."))
    return names


ENTRY_POINTS = _entry_point_modules()


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_main_takes_argv_so_a_test_never_has_to_patch_sys_argv(name: str) -> None:
    signature = inspect.signature(importlib.import_module(name).main)

    assert "argv" in signature.parameters


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_main_injects_its_corpus_so_a_test_never_needs_the_real_one(name: str) -> None:
    """Loading BHSA is the one dependency a test cannot supply as a file, so it is a parameter."""
    signature = inspect.signature(importlib.import_module(name).main)

    assert signature.parameters["corpus_factory"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_the_injected_corpus_factory_defaults_to_the_real_loader(name: str) -> None:
    default = (
        inspect.signature(importlib.import_module(name).main).parameters["corpus_factory"].default
    )

    assert callable(default)


@pytest.mark.parametrize("name", ENTRY_POINTS)
@pytest.mark.parametrize("option", SHARED_OPTIONS)
def test_no_entry_point_declares_a_shared_option_itself(name: str, option: str) -> None:
    module = importlib.import_module(name)
    source = inspect.getsource(module)

    assert f'add_argument("{option}"' not in source


def test_every_entry_point_was_discovered() -> None:
    """A shrinking list would silently stop holding the contract, so the count is asserted."""
    assert len(ENTRY_POINTS) == 27
