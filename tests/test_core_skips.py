"""The skip line and its reader agree, so a logged skip explains a missing dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.skips import report_skip, skipped_in_log


def test_the_reported_line_is_what_the_reader_finds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_skip("c_icf", "retrieval scoring needs at least two pairs")
    log = tmp_path / "x.log"
    log.write_text("scoring...\n" + capsys.readouterr().err)
    assert skipped_in_log(log) == {"c_icf"}


def test_a_missing_log_reports_nothing_skipped(tmp_path: Path) -> None:
    assert skipped_in_log(tmp_path / "none.log") == set()
