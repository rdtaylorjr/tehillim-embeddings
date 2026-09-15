# One command regenerates every declared partition: `snakemake -j 8 --rerun-triggers params`.
import sys
from pathlib import Path

HERE = Path(str(workflow.current_basedir))
PYTHON = str(HERE / ".venv" / "bin" / "python")
sys.path.insert(0, str(HERE / "src"))

from core.driver import default_plan, provenance_of, render_rules

DATA_ROOT = Path(config.get("data_root", "data"))
CONFIG_ROOT = Path(config.get("config_root", "config"))
GPU = bool(int(config.get("gpu", 0)))
GPU_FLAG = " --gpu" if GPU else ""

PLAN = default_plan(DATA_ROOT, CONFIG_ROOT, gpu=GPU)
CELLS_BY_NAME = {cell.name: cell for cell in PLAN.runnable}


def provenance_at_run(name):
    """Hashes a cell's code, arguments and inputs when Snakemake schedules it, inputs regenerated."""
    return provenance_of(CELLS_BY_NAME[name])


RULES = HERE / ".snakemake" / "cells.smk"
RULES.parent.mkdir(exist_ok=True)
ROOTS = f"--data-root {DATA_ROOT} --config-root {CONFIG_ROOT}{GPU_FLAG}"
RULES.write_text(
    render_rules(PLAN.runnable, python=PYTHON, provenance="provenance_at_run", roots=ROOTS)
)

include: str(RULES)


# No declared output, so the manifest step runs on every invocation and demands every cell's outputs.
rule all:
    input:
        [str(p) for cell in PLAN.runnable for p in cell.outputs],
    shell:
        f"{PYTHON} -m core.driver manifest --data-root {DATA_ROOT} --config-root {CONFIG_ROOT}{GPU_FLAG}"
