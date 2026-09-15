"""The line a batch writes when it passes over a dataset, and the reader parity checks share."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_SKIP_LINE = re.compile(r"^skipping (\S+):", re.MULTILINE)


def report_skip(identifier: str, reason: str) -> None:
    """Writes the skip line for one dataset to stderr, where a cell's log captures it."""
    print(f"skipping {identifier}: {reason}", file=sys.stderr)


def skipped_in_log(log: Path) -> set[str]:
    """The dataset names a stage reported skipping, read from its captured log."""
    if not log.exists():
        return set()
    return set(_SKIP_LINE.findall(log.read_text()))
