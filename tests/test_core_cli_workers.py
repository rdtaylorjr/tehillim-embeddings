"""The shared worker-count argument."""

from __future__ import annotations

import argparse

from core.cli import add_workers_argument
from core.parallel import default_max_workers


def test_workers_default_to_the_machine_and_take_an_explicit_count() -> None:
    parser = argparse.ArgumentParser()
    add_workers_argument(parser)
    assert parser.parse_args([]).workers == default_max_workers()
    assert parser.parse_args(["--workers", "3"]).workers == 3
