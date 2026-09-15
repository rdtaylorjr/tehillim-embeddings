"""Content and code identity for driver cells, and the sidecar manifest each cell writes."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from core.spec import FAMILY_PACKAGES

#: Every package whose source is part of a generator's code identity.
REPO_PACKAGES: tuple[str, ...] = (*FAMILY_PACKAGES, "core", "families")
SRC_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "_manifest.json"


def file_hash(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def in_repo_imports(source: str, packages: tuple[str, ...]) -> set[str]:
    """Dotted module names a source file imports from the named packages."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names = [node.module]
        else:
            continue
        found.update(n for n in names if n.split(".")[0] in packages)
    return found


def _module_file(module: str, src_roots: tuple[Path, ...]) -> Path | None:
    """The source file a dotted module name resolves to under the first root holding it."""
    for root in src_roots:
        path = root.joinpath(*module.split("."))
        candidate = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        if candidate.exists():
            return candidate
    return None


def code_hash(
    module: str,
    src_roots: Path | tuple[Path, ...] = SRC_ROOT,
    packages: tuple[str, ...] = REPO_PACKAGES,
) -> str:
    """SHA-256 over the module's source and every module it reaches by import under the roots."""
    roots = (src_roots,) if isinstance(src_roots, Path) else src_roots
    pending = [module]
    files: dict[str, Path] = {}
    while pending:
        name = pending.pop()
        path = _module_file(name, roots)
        if name in files or path is None:
            continue
        files[name] = path
        pending.extend(in_repo_imports(path.read_text(), packages))
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode())
        digest.update(files[name].read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Manifest:
    """What one cell read, what code ran, and what it wrote."""

    cell: str
    rule: str
    inputs: dict[str, str]
    code_hash: str
    parameters: dict[str, str]
    outputs: dict[str, str]
    repository_revision: str
    started: str
    duration_s: float

    def to_dict(self) -> dict[str, object]:
        """The JSON-serialisable form."""
        return asdict(self)


def _records_in(path: Path) -> dict[str, object]:
    """The sidecar's records by cell, or none where the file is absent or predates that shape."""
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict) or "cell" in loaded:
        return {}
    return {str(cell): record for cell, record in loaded.items()}


def write_manifest(manifest: Manifest, directory: Path) -> Path:
    """Records the cell in the directory's sidecar, keeping the other cells that wrote there."""
    path = directory / MANIFEST_NAME
    records = _records_in(path)
    records[manifest.cell] = manifest.to_dict()
    path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    return path
