"""Plans generator cells from the declared specs, renders them as rules, and runs one cell."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from core.cli import add_config_root_argument
from core.provenance import Manifest, code_hash, file_hash, write_manifest
from core.spec import (
    GeneratorSpec,
    SupportSpec,
    discover_specs,
    discover_support_specs,
    partition_paths,
    support_paths,
)
from semantic.api_models import API_KEY_ENV_VARS

Resource = str | None


@dataclass(frozen=True, slots=True)
class Cell:
    """One unit of work: a module invocation with known inputs, outputs and resource."""

    name: str
    module: str
    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]
    command_args: list[str]
    resource: Resource


@dataclass(frozen=True, slots=True)
class Plan:
    """Every cell the tree calls for, split by whether this machine can run it."""

    runnable: tuple[Cell, ...]
    blocked: tuple[Cell, ...]

    @property
    def expected_outputs(self) -> tuple[Path, ...]:
        """Every output any cell owes, runnable or not."""
        return tuple(p for cell in (*self.runnable, *self.blocked) for p in cell.outputs)


def cell_for(spec: GeneratorSpec, data_root: Path, config_root: Path) -> Cell:
    """The cell a generator spec describes, with the args that select its variant."""
    args = ["--output-root", str(data_root)]
    if spec.support:
        args += ["--config-root", str(config_root)]
    args += list(spec.args)
    return Cell(
        name=spec.name,
        module=spec.module,
        inputs=support_paths(spec, config_root),
        outputs=partition_paths(spec, data_root),
        command_args=args,
        resource=spec.resource,
    )


def cells_for_support(specs: Sequence[SupportSpec], config_root: Path) -> list[Cell]:
    """The cells that write the frozen support tables, which read only BHSA."""
    return [
        Cell(
            name=spec.module,
            module=spec.module,
            inputs=(),
            outputs=tuple(config_root / name for name in spec.outputs),
            command_args=["--config-root", str(config_root)],
            resource=None,
        )
        for spec in specs
    ]


def available_resources(env: Mapping[str, str], *, gpu: bool) -> set[Resource]:
    """Resources this run may hold: `api` when every provider key is set, `gpu` when declared."""
    available: set[Resource] = {None}
    if all(env.get(var) for var in API_KEY_ENV_VARS.values()):
        available.add("api")
    if gpu:
        available.add("gpu")
    return available


def plan(
    specs: Sequence[GeneratorSpec],
    supports: Sequence[SupportSpec],
    data_root: Path,
    config_root: Path,
    *,
    available: set[Resource],
) -> Plan:
    """Turns the declarations into cells and separates those this machine cannot run."""
    cells = cells_for_support(supports, config_root) + [
        cell_for(spec, data_root, config_root) for spec in specs
    ]
    runnable = tuple(c for c in cells if c.resource in available)
    blocked = tuple(c for c in cells if c.resource not in available)
    return Plan(runnable=runnable, blocked=blocked)


#: Options that change how a cell runs, never what it produces, so they stay out of its identity.
EXECUTION_OPTIONS: frozenset[str] = frozenset({"--workers"})


def result_arguments(cell: Cell) -> list[str]:
    """The cell's arguments without the execution-only options and their values."""
    kept: list[str] = []
    skip = False
    for argument in cell.command_args:
        if skip:
            skip = False
            continue
        if argument in EXECUTION_OPTIONS:
            skip = True
            continue
        kept.append(argument)
    return kept


def provenance_of(cell: Cell, code_hash_of: Callable[[str], str] = code_hash) -> str:
    """The params string a rule carries: code identity, the cell's arguments, every input's hash."""
    record = {"code": code_hash_of(cell.module), "args": " ".join(result_arguments(cell))}
    record.update({str(p): file_hash(p) for p in cell.inputs if p.exists()})
    return json.dumps(record, sort_keys=True)


def _rule_name(cell: Cell) -> str:
    """A Snakemake-safe rule name for a cell."""
    return "cell__" + cell.name.replace(".", "_").replace("-", "_")


def render_rules(
    cells: Sequence[Cell],
    *,
    python: str,
    provenance: str,
    roots: str = "",
    runner: str = "core.driver",
    threads: int = 1,
) -> str:
    """Renders one rule per cell, invoking the named runner and carrying its provenance."""
    blocks = []
    for cell in cells:
        lines = [f"rule {_rule_name(cell)}:"]
        if cell.inputs:
            lines.append("    input:")
            lines.append("        " + ",\n        ".join(repr(str(p)) for p in cell.inputs) + ",")
        lines.append("    output:")
        lines.append("        " + ",\n        ".join(repr(str(p)) for p in cell.outputs) + ",")
        lines.append("    params:")
        #: Snakemake evaluates a wildcards-only callable at scheduling, after inputs regenerate.
        params = f"lambda wildcards, name={cell.name!r}: {provenance}(name)"
        lines.append(f"        provenance={params},")
        if cell.resource is not None:
            lines.append("    resources:")
            lines.append(f"        {cell.resource}=1,")
        lines.append(f"    threads: {threads}")
        lines.append(f"    log: 'logs/{cell.name}.log'")
        lines.append("    shell:")
        lines.append(
            f"        '{python} -m {runner} run --cell {cell.name} {roots} > {{log}} 2>&1'"
        )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def _git_revision(run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> str:
    """The checked-out commit, read with the one read-only git command the repo allows."""
    result = run(["git", "log", "-1", "--format=%H"], capture_output=True, text=True, check=False)
    return result.stdout.strip() or "unknown"


def _module_main(module: str) -> Callable[[list[str]], None]:
    """The generator's main, imported at call time so planning never loads a model stack."""
    main: Callable[[list[str]], None] = importlib.import_module(module).main
    return main


def run_cell(
    cell: Cell,
    *,
    main: Callable[[list[str]], None] | None = None,
    revision: Callable[[], str] = _git_revision,
    code_hash_of: Callable[[str], str] = code_hash,
) -> None:
    """Runs one cell's generator, checks every output exists, and writes the sidecar manifests."""
    started = datetime.now(UTC)
    (main or _module_main(cell.module))(cell.command_args)
    for output in cell.outputs:
        if not output.exists():
            msg = f"{cell.name} finished without writing {output}"
            raise FileNotFoundError(msg)
    manifest = Manifest(
        cell=cell.name,
        rule=_rule_name(cell),
        inputs={str(p): file_hash(p) for p in cell.inputs},
        code_hash=code_hash_of(cell.module),
        parameters={"args": " ".join(cell.command_args)},
        outputs={str(p): file_hash(p) for p in cell.outputs},
        repository_revision=revision(),
        started=started.isoformat(),
        duration_s=(datetime.now(UTC) - started).total_seconds(),
    )
    for directory in sorted({p.parent for p in cell.outputs}):
        write_manifest(manifest, directory)


class RunIncompleteError(RuntimeError):
    """Raised when an expected output is absent after every runnable cell has run."""


def write_run_manifest(
    result: Plan, data_root: Path, *, revision: Callable[[], str] = _git_revision
) -> Path:
    """Writes the run-level manifest and raises when any expected output is missing."""
    expected = result.expected_outputs
    present = [str(p) for p in expected if p.exists()]
    missing = [str(p) for p in expected if not p.exists()]
    record = {
        "finished": datetime.now(UTC).isoformat(),
        "repository_revision": revision(),
        "expected_cells": len(result.runnable) + len(result.blocked),
        "complete": present,
        "missing": missing,
        "blocked": {cell.name: cell.resource for cell in result.blocked},
    }
    path = data_root / "_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    if missing:
        names = sorted(c.name for c in result.blocked if any(not p.exists() for p in c.outputs))
        msg = f"expected outputs missing from blocked cells {names}: {missing}"
        raise RunIncompleteError(msg)
    return path


def default_plan(data_root: Path, config_root: Path, *, gpu: bool) -> Plan:
    """The plan for the shipped generators on this machine."""
    return plan(
        discover_specs(),
        discover_support_specs(),
        data_root,
        config_root,
        available=available_resources(os.environ, gpu=gpu),
    )


def main(
    argv: list[str] | None = None,
    *,
    plan_factory: Callable[..., Plan] = default_plan,
) -> None:
    """`run --cell NAME` executes one planned cell, `manifest` writes the run-level record."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "manifest"):
        command = sub.add_parser(name)
        command.add_argument("--data-root", type=Path, default=Path("data"))
        add_config_root_argument(command)
        command.add_argument("--gpu", action="store_true")
        if name == "run":
            command.add_argument("--cell", required=True)
    args = parser.parse_args(argv)
    cells = plan_factory(args.data_root, args.config_root, gpu=args.gpu)
    if args.command == "manifest":
        write_run_manifest(cells, args.data_root)
        return
    by_name = {c.name: c for c in (*cells.runnable, *cells.blocked)}
    if args.cell not in by_name:
        parser.error(f"unknown cell {args.cell}")
    run_cell(by_name[args.cell])


if __name__ == "__main__":
    main()
