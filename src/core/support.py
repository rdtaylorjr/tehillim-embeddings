"""External-support thresholding: rare-signature collapsing and vocabulary construction."""

from __future__ import annotations

import csv
from pathlib import Path

RARE_TOKEN = "<RARE>"


def load_external_signature_counts(path: Path) -> dict[str, int]:
    """Reads a `signature,count` CSV into a lookup table."""
    with path.open(newline="") as handle:
        return {row["signature"]: int(row["count"]) for row in csv.DictReader(handle)}


def collapse_rare(signature: str, external_counts: dict[str, int], k: int) -> str:
    """`signature` unchanged if its whole-Bible-outside-Psalms count is >= k, else `<RARE>`."""
    return signature if external_counts.get(signature, 0) >= k else RARE_TOKEN


def build_signature_vocabulary(external_counts: dict[str, int], k: int) -> tuple[str, ...]:
    """Every signature at or above `k`, sorted, plus the `<RARE>` token last."""
    surviving = sorted(signature for signature, count in external_counts.items() if count >= k)
    return (*surviving, RARE_TOKEN)


def write_external_signature_counts(path: Path, counts: dict[str, int]) -> None:
    """Writes the `signature,count` CSV load_external_signature_counts reads, densest first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["signature", "count"])
        #: Descending count then name, so a vocabulary's keepers head the file and reruns match.
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        writer.writerows(ranked)


def collapsed_sequences(
    sequences: tuple[tuple[str, ...], ...], external_counts: dict[str, int], k: int
) -> tuple[tuple[str, ...], ...]:
    """Per-node sequences with every sub-threshold value collapsed to RARE."""
    return tuple(
        tuple(collapse_rare(value, external_counts, k) for value in node_values)
        for node_values in sequences
    )
