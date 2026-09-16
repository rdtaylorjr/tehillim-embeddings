"""One definition of where a construction is written and when an existing one is skipped."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.export import path_to_write
from core.partition import BHSA_HALF_VERSE, Partition

TYP_ICF = Partition(BHSA_HALF_VERSE, "syntactic", level="clause", feature="typ", construction="icf")


class TestPathToWrite:
    def test_returns_the_path_the_construction_belongs_at(self, tmp_path: Path) -> None:
        assert path_to_write(tmp_path, TYP_ICF) == TYP_ICF.file(tmp_path)

    def test_returns_none_once_the_dataset_is_already_written(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A rerun must skip completed work rather than recompute and rewrite it."""
        path = TYP_ICF.file(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

        assert path_to_write(tmp_path, TYP_ICF) is None
        assert capsys.readouterr().err == ""

    def test_announces_the_partition_it_is_about_to_compute(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path_to_write(tmp_path, TYP_ICF)

        assert TYP_ICF.directory in capsys.readouterr().err

    def test_carries_every_partition_key_through_to_the_path(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = path_to_write(tmp_path, TYP_ICF)
        capsys.readouterr()

        assert path is not None
        assert path.relative_to(tmp_path).as_posix() == (
            "corpus=bhsa/unit=half_verse/domain=syntactic/level=clause/feature=typ/"
            "construction=icf/part-0.parquet"
        )
