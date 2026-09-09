"""One definition of where a construction is written and when an existing one is skipped."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.export import dataset_path, path_to_write


class TestPathToWrite:
    def test_returns_the_path_the_construction_belongs_at(self, tmp_path: Path) -> None:
        path = path_to_write(tmp_path, "typ", "icf", domain="syntactic", unit_key="feature")

        assert path == dataset_path(tmp_path, "typ", "icf", domain="syntactic", unit_key="feature")

    def test_returns_none_once_the_dataset_is_already_written(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A rerun must skip completed work rather than recompute and rewrite it."""
        path = dataset_path(tmp_path, "typ", "icf", domain="syntactic", unit_key="feature")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

        assert path_to_write(tmp_path, "typ", "icf", domain="syntactic", unit_key="feature") is None
        assert capsys.readouterr().err == ""

    def test_announces_only_the_construction_it_is_about_to_compute(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path_to_write(tmp_path, "typ", "icf", domain="syntactic", unit_key="feature")

        assert "syntactic feature=typ construction=icf" in capsys.readouterr().err

    def test_carries_the_optional_partition_tiers_through_to_the_path(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = path_to_write(
            tmp_path, "typ", "icf", domain="syntactic", unit_key="feature", level="phrase"
        )
        capsys.readouterr()

        assert "level=phrase" in str(path)
