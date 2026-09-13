"""Check 2 of the driver design: every entry point is owned by exactly one declared spec."""

import re
import tomllib
from pathlib import Path

from core.spec import discover_specs, discover_support_specs

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
#: Entry points that are the driver itself rather than cells it plans, with the reason.
DRIVER_ENTRY_POINTS = {"core.driver": "the cell runner every rendered rule invokes"}


def _modules_with_main() -> set[str]:
    """Dotted names of every shipped module defining a main()."""
    found = set()
    for path in SRC.rglob("*.py"):
        if "egg-info" in path.parts:
            continue
        if re.search(r"^def main\(", path.read_text(), re.MULTILINE):
            found.add(".".join(path.relative_to(SRC).with_suffix("").parts))
    return found


def _console_script_modules() -> set[str]:
    """Modules behind every [project.scripts] entry."""
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"].get("scripts", {})
    return {target.split(":")[0] for target in scripts.values()}


def _planned_modules() -> set[str]:
    """Modules the driver plans over, from the generator and support declarations."""
    generators = {spec.module.split(":")[0] for spec in discover_specs()}
    builders = {spec.module for spec in discover_support_specs()}
    return generators | builders


def test_every_main_is_planned_by_a_spec() -> None:
    """A module with main() and no spec would be runnable by hand and invisible to the driver."""
    assert _modules_with_main() - _planned_modules() - set(DRIVER_ENTRY_POINTS) == set()


def test_every_console_script_is_planned_by_a_spec() -> None:
    """An installed command that the driver never invokes is a second way to run the pipeline."""
    assert _console_script_modules() - _planned_modules() == set()


def test_every_planned_module_has_a_main() -> None:
    """A spec that names a module without main() cannot be run as a cell."""
    assert _planned_modules() - _modules_with_main() == set()
