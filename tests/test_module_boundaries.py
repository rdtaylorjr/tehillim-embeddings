"""Holds the source tree to its import boundaries: no domain reaches into another's internals."""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
DOMAINS = ("core", "lexical", "morphology", "semantic", "syntax")


def _modules() -> list[pathlib.Path]:
    return sorted(p for p in SRC.rglob("*.py") if "egg-info" not in p.parts)


def _imports(path: pathlib.Path) -> list[tuple[str, str]]:
    """Every (module, imported name) pair a file brings in from the source tree."""
    tree = ast.parse(path.read_text())
    return [
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
        for alias in node.names
        if node.module.split(".")[0] in DOMAINS
    ]


@pytest.mark.parametrize("path", _modules(), ids=lambda p: str(p.relative_to(SRC)))
def test_no_module_imports_another_modules_private_name(path: pathlib.Path) -> None:
    """A leading underscore means module-internal, so importing one couples across a boundary."""
    private = [f"{module}.{name}" for module, name in _imports(path) if name.startswith("_")]

    assert private == []


@pytest.mark.parametrize("path", _modules(), ids=lambda p: str(p.relative_to(SRC)))
def test_no_representation_domain_imports_another_representation_domain(
    path: pathlib.Path,
) -> None:
    """Only core is shared; a domain reaching sideways would couple two representations."""
    domain = path.relative_to(SRC).parts[0]
    if domain == "core":
        return
    sideways = sorted(
        {
            module.split(".")[0]
            for module, _ in _imports(path)
            if module.split(".")[0] not in ("core", domain)
        }
    )

    assert sideways == []


def test_a_type_name_is_defined_once_across_the_shared_core() -> None:
    """Two different types under one name in core is a collision waiting to be imported wrong."""
    defined: dict[str, list[str]] = {}
    for path in sorted((SRC / "core").glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                defined.setdefault(node.name, []).append(path.name)
            elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name[0].isupper():
                    defined.setdefault(name, []).append(path.name)

    assert {name: files for name, files in defined.items() if len(files) > 1} == {}


@pytest.mark.parametrize("path", _modules(), ids=lambda p: str(p.relative_to(SRC)))
def test_every_multi_argument_zip_is_strict(path: pathlib.Path) -> None:
    """The parallel per-half-verse tuples must misalign loudly, never silently truncate."""
    loose = [
        node.lineno
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "zip"
        and len(node.args) > 1
        and not any(keyword.arg == "strict" for keyword in node.keywords)
    ]

    assert loose == []
