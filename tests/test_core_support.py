"""Round-trips the `signature,count` support format its writer and reader both define."""

from __future__ import annotations

from pathlib import Path

from core.support import load_external_signature_counts, write_external_signature_counts


def test_external_signature_counts_round_trip_through_the_csv(tmp_path: Path) -> None:
    """The writer and the reader are one format, so what one writes the other must read back."""
    path = tmp_path / "support.csv"
    counts = {"A": 7, "B": 2, "C": 7}

    write_external_signature_counts(path, counts)

    assert load_external_signature_counts(path) == counts


def test_external_signature_counts_are_written_densest_first(tmp_path: Path) -> None:
    """Ordering by descending count puts the signatures a vocabulary will keep at the top."""
    path = tmp_path / "support.csv"

    write_external_signature_counts(path, {"rare": 1, "common": 9, "mid": 4})

    assert [line.split(",")[0] for line in path.read_text().splitlines()[1:]] == [
        "common",
        "mid",
        "rare",
    ]


def test_signatures_sharing_a_count_are_written_in_name_order(tmp_path: Path) -> None:
    """A tie broken by name keeps the file byte-identical across runs."""
    path = tmp_path / "support.csv"

    write_external_signature_counts(path, {"b": 5, "a": 5})

    assert [line.split(",")[0] for line in path.read_text().splitlines()[1:]] == ["a", "b"]
