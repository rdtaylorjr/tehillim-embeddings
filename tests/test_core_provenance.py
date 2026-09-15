import json
from pathlib import Path

from core.provenance import (
    Manifest,
    code_hash,
    file_hash,
    in_repo_imports,
    write_manifest,
)


class TestFileHash:
    def test_is_content_based_and_stable(self, tmp_path: Path) -> None:
        """Two files with the same bytes hash alike, a changed byte changes the hash."""
        a, b = tmp_path / "a", tmp_path / "b"
        a.write_bytes(b"xyz")
        b.write_bytes(b"xyz")
        assert file_hash(a) == file_hash(b)
        b.write_bytes(b"xyw")
        assert file_hash(a) != file_hash(b)


class TestInRepoImports:
    def test_collects_only_imports_of_the_named_packages(self, tmp_path: Path) -> None:
        """Third-party imports are outside the code identity, repo packages are inside it."""
        source = (
            "import numpy\nfrom core.spec import x\nimport syntactic.corpus\nfrom os import path\n"
        )
        assert in_repo_imports(source, ("core", "syntactic")) == {"core.spec", "syntactic.corpus"}


class TestCodeHash:
    def test_follows_imports_transitively_within_the_repo(self, tmp_path: Path) -> None:
        """Changing a module two imports away changes the identity of the caller."""
        src = tmp_path / "src"
        (src / "pkg").mkdir(parents=True)
        (src / "pkg" / "__init__.py").write_text("")
        (src / "pkg" / "a.py").write_text("from pkg.b import f\n")
        (src / "pkg" / "b.py").write_text("from pkg.c import g\n")
        (src / "pkg" / "c.py").write_text("g = 1\n")
        before = code_hash("pkg.a", src, ("pkg",))
        (src / "pkg" / "c.py").write_text("g = 2\n")
        assert code_hash("pkg.a", src, ("pkg",)) != before

    def test_is_independent_of_modules_not_reached(self, tmp_path: Path) -> None:
        """An unrelated module changing leaves the identity alone."""
        src = tmp_path / "src"
        (src / "pkg").mkdir(parents=True)
        (src / "pkg" / "__init__.py").write_text("")
        (src / "pkg" / "a.py").write_text("x = 1\n")
        (src / "pkg" / "z.py").write_text("y = 1\n")
        before = code_hash("pkg.a", src, ("pkg",))
        (src / "pkg" / "z.py").write_text("y = 2\n")
        assert code_hash("pkg.a", src, ("pkg",)) == before

    def test_resolves_the_real_generators(self) -> None:
        """Every planned module hashes without error against the shipped tree."""
        from core.spec import discover_specs

        for spec in discover_specs():
            assert len(code_hash(spec.module.split(":")[0])) == 64


class TestManifest:
    def test_round_trips_through_json_beside_the_outputs(self, tmp_path: Path) -> None:
        """The sidecar records what was read, what code ran, and what was written."""
        out = tmp_path / "domain=x/unit=y/construction=z/part-0.parquet"
        out.parent.mkdir(parents=True)
        out.write_bytes(b"data")
        manifest = Manifest(
            cell="domain=x/unit=y/construction=z",
            rule="gen",
            inputs={"config/s.csv": "abc"},
            code_hash="def",
            parameters={"k": "1000"},
            outputs={str(out): file_hash(out)},
            repository_revision="rev",
            started="2026-09-12T00:00:00+00:00",
            duration_s=1.5,
        )
        path = write_manifest(manifest, out.parent)
        assert path.name == "_manifest.json"
        assert json.loads(path.read_text()) == {manifest.cell: manifest.to_dict()}

    def test_cells_sharing_a_directory_each_keep_their_record(self, tmp_path: Path) -> None:
        """Several cells write one interface directory, so the sidecar is a map by cell."""
        first = Manifest("a", "gen", {}, "h1", {}, {}, "rev", "t", 1.0)
        second = Manifest("b", "gen", {}, "h2", {}, {}, "rev", "t", 2.0)
        rerun = Manifest("a", "gen", {}, "h3", {}, {}, "rev", "t", 3.0)
        for manifest in (first, second, rerun):
            path = write_manifest(manifest, tmp_path)
        record = json.loads(path.read_text())
        assert set(record) == {"a", "b"}
        assert record["a"]["code_hash"] == "h3"
        assert record["b"]["code_hash"] == "h2"

    def test_a_sidecar_that_is_not_a_map_by_cell_is_replaced(self, tmp_path: Path) -> None:
        (tmp_path / "_manifest.json").write_text('{"cell": "old", "code_hash": "x"}')
        path = write_manifest(Manifest("a", "gen", {}, "h", {}, {}, "rev", "t", 1.0), tmp_path)
        assert set(json.loads(path.read_text())) == {"a"}


class TestCodeHashAcrossRoots:
    def test_follows_an_import_into_a_second_source_root(self, tmp_path: Path) -> None:
        """A consumer repository's identity includes the producer modules it imports."""
        a, b = tmp_path / "a", tmp_path / "b"
        (a / "app").mkdir(parents=True)
        (b / "lib").mkdir(parents=True)
        (a / "app" / "__init__.py").write_text("")
        (a / "app" / "run.py").write_text("from lib.core import f\n")
        (b / "lib" / "__init__.py").write_text("")
        (b / "lib" / "core.py").write_text("f = 1\n")
        before = code_hash("app.run", (a, b), ("app", "lib"))
        (b / "lib" / "core.py").write_text("f = 2\n")
        assert code_hash("app.run", (a, b), ("app", "lib")) != before
