"""Computes whole-Bible-outside-Psalms typ:function:det signature frequencies (H5.8's S+det)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

from phrase.corpus import Corpus
from phrase.signature import build_full_phrase_signature

_PSALMS_BOOK_NAME = "Psalmi"


def build_external_full_signature_counts(api: Any) -> dict[str, int]:
    """Tallies `build_full_phrase_signature(...)` over every phrase atom outside Psalms."""
    F, L = api.F, api.L  # noqa: N806
    counts: dict[str, int] = {}
    for book in F.otype.s("book"):
        if F.book.v(book) == _PSALMS_BOOK_NAME:
            continue
        for atom in L.d(book, otype="phrase_atom"):
            mother = L.u(atom, otype="phrase")
            signature = build_full_phrase_signature(
                typ=F.typ.v(atom), function=F.function.v(mother[0]), det=F.det.v(atom)
            )
            counts[signature] = counts.get(signature, 0) + 1
    return counts


def main() -> None:
    """Writes `data/config/phrase_full_signature_external_support.csv`."""
    parser_output = Path(__file__).resolve().parents[3] / "data" / "config"
    parser_output.mkdir(parents=True, exist_ok=True)
    output_path = parser_output / "phrase_full_signature_external_support.csv"

    corpus = Corpus.load()
    counts = build_external_full_signature_counts(corpus.api)

    with open(output_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["signature", "count"])
        for signature, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([signature, count])

    print(f"wrote {len(counts)} distinct signatures to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
