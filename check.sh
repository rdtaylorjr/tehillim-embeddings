#!/usr/bin/env bash
# Every gate CI runs, in CI's order, so a local pass cannot diverge from a CI pass.
set -euo pipefail
PY=.venv/bin/python
$PY -m ruff check .
$PY -m ruff format --check .
$PY -m mypy --strict src
$PY -m deptry src
$PY -m vulture src .vulture-whitelist.py --min-confidence 60
$PY -m pytest -q -m "not integration and not api_integration"
