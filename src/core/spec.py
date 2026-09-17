"""Static declaration of what each generator module writes, so a driver can plan without running."""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from pathlib import Path

from core.partition import Partition

#: The packages whose generate* modules the driver plans over.
FAMILY_PACKAGES: tuple[str, ...] = ("lexical", "morphological", "syntactic", "semantic")


@dataclass(frozen=True, slots=True)
class GeneratorSpec:
    """One generator: the partitions it writes, the support tables it reads, its resource."""

    module: str
    partitions: tuple[Partition, ...]
    support: tuple[str, ...] = ()
    resource: str | None = None
    #: One module can declare several cells: the variant names the cell, the args select it.
    variant: str = ""
    args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """A spec that writes nothing or writes one path twice is a declaration error."""
        if not self.partitions:
            msg = f"{self.name}: no partitions declared"
            raise ValueError(msg)
        if len(set(self.partitions)) != len(self.partitions):
            msg = f"{self.name}: duplicate partitions declared"
            raise ValueError(msg)
        if bool(self.variant) != bool(self.args):
            msg = f"{self.name}: a variant and its args are declared together or not at all"
            raise ValueError(msg)

    @property
    def name(self) -> str:
        """The cell's name: the module, and the variant where the module declares several."""
        return f"{self.module}.{self.variant}" if self.variant else self.module


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


def partition_paths(spec: GeneratorSpec, root: Path) -> tuple[Path, ...]:
    """The parquet file each declared partition is written at under the root."""
    return tuple(partition.file(root) for partition in spec.partitions)


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
