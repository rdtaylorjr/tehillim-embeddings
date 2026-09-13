"""Static declaration of what each generator module writes, so a driver can plan without running."""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from pathlib import Path

#: The packages whose generate* modules the driver plans over.
FAMILY_PACKAGES: tuple[str, ...] = ("lexical", "morphological", "syntactic", "semantic")
PART_FILE = "part-0.parquet"


@dataclass(frozen=True, slots=True)
class GeneratorSpec:
    """One generator: the partitions it writes, the support tables it reads, its resource."""

    module: str
    partitions: tuple[str, ...]
    support: tuple[str, ...] = ()
    resource: str | None = None

    def __post_init__(self) -> None:
        """A spec that writes nothing or writes one path twice is a declaration error."""
        if not self.partitions:
            msg = f"{self.module}: no partitions declared"
            raise ValueError(msg)
        if len(set(self.partitions)) != len(self.partitions):
            msg = f"{self.module}: duplicate partitions declared"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SupportSpec:
    """One support-table builder and the config tables it writes."""

    module: str
    outputs: tuple[str, ...]

    def __post_init__(self) -> None:
        """A builder that writes nothing is a declaration error."""
        if not self.outputs:
            msg = f"{self.module}: no outputs declared"
            raise ValueError(msg)


def partition_dirs(
    domain: str,
    unit_key: str,
    unit: str,
    constructions: tuple[str, ...],
    *,
    level: str | None = None,
    text: str | None = None,
) -> tuple[str, ...]:
    """Hive-relative directories in the tree's segment order, one per construction."""
    head = [f"domain={domain}"]
    if level is not None:
        head.append(f"level={level}")
    head.append(f"{unit_key}={unit}")
    if text is not None:
        head.append(f"text={text}")
    if not constructions:
        return ("/".join(head),)
    return tuple("/".join([*head, f"construction={c}"]) for c in constructions)


def partition_paths(spec: GeneratorSpec, root: Path) -> tuple[Path, ...]:
    """The parquet file each declared partition is written at under the root."""
    return tuple(root / partition / PART_FILE for partition in spec.partitions)


def discover_specs(packages: tuple[str, ...] = FAMILY_PACKAGES) -> list[GeneratorSpec]:
    """Imports every generate* module in the family packages and collects its declarations."""
    specs: list[GeneratorSpec] = []
    for package_name in packages:
        package = importlib.import_module(package_name)
        for info in pkgutil.iter_modules(package.__path__):
            if not info.name.startswith("generate"):
                continue
            module = importlib.import_module(f"{package_name}.{info.name}")
            declared = getattr(module, "SPECS", None)
            if declared is None:
                declared = (module.SPEC,)
            specs.extend(declared)
    return specs


def support_paths(spec: GeneratorSpec, config_root: Path) -> tuple[Path, ...]:
    """The frozen support tables a generator reads, under the config root."""
    return tuple(config_root / filename for filename in spec.support)


def discover_support_specs(packages: tuple[str, ...] = FAMILY_PACKAGES) -> list[SupportSpec]:
    """Imports every compute_* module in the family script packages and collects its declaration."""
    specs: list[SupportSpec] = []
    for package_name in packages:
        try:
            scripts = importlib.import_module(f"{package_name}.scripts")
        except ModuleNotFoundError:
            continue
        for info in pkgutil.iter_modules(scripts.__path__):
            if not info.name.startswith("compute_"):
                continue
            module = importlib.import_module(f"{package_name}.scripts.{info.name}")
            specs.append(module.SPEC)
    return specs
