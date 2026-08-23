"""Frozen, label-blind minimum-support threshold for the phrase_signature vocabulary."""

from __future__ import annotations

import csv
from pathlib import Path

RARE_TOKEN = "<RARE>"

# Frozen 2026-08-22 from the external-support curve alone, before any Phase 5C benchmark run.
MIN_EXTERNAL_SUPPORT_K = 1000

# Frozen 2026-08-22 from the S+det external-support curve alone, before any Phase 5D benchmark run.
MIN_EXTERNAL_SUPPORT_K_FULL = 1000


def load_external_signature_counts(path: Path) -> dict[str, int]:
    """Reads a `signature,count` CSV into a lookup table."""
    with open(path, newline="") as handle:
        return {row["signature"]: int(row["count"]) for row in csv.DictReader(handle)}


def collapse_rare(signature: str, external_counts: dict[str, int], k: int) -> str:
    """`signature` unchanged if its whole-Bible-outside-Psalms count is >= k, else `<RARE>`."""
    return signature if external_counts.get(signature, 0) >= k else RARE_TOKEN


def build_signature_vocabulary(external_counts: dict[str, int], k: int) -> tuple[str, ...]:
    """Every signature at or above `k`, sorted, plus the `<RARE>` token last."""
    surviving = sorted(signature for signature, count in external_counts.items() if count >= k)
    return (*surviving, RARE_TOKEN)
