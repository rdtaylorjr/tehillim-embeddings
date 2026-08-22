"""Computes whole-Bible-outside-Psalms signature frequencies: the frozen, label-blind support."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

from morphological.corpus import Corpus
from morphological.signature import build_signature

_PSALMS_BOOK_NAME = "Psalmi"


def build_external_signature_counts(api: Any) -> dict[str, int]:
    """Tallies `build_signature(...)` over every word outside the Psalms book."""
    F, L = api.F, api.L  # noqa: N806
    counts: dict[str, int] = {}
    for book in F.otype.s("book"):
        if F.book.v(book) == _PSALMS_BOOK_NAME:
            continue
        for word in L.d(book, otype="word"):
            signature = build_signature(
                sp=F.sp.v(word),
                gn=F.gn.v(word),
                nu=F.nu.v(word),
                ps=F.ps.v(word),
                st=F.st.v(word),
                vs=F.vs.v(word),
                vt=F.vt.v(word),
            )
            counts[signature] = counts.get(signature, 0) + 1
    return counts


def main() -> None:
    """Writes `data/config/morph_signature_external_support.csv`."""
    parser_output = Path(__file__).resolve().parents[3] / "data" / "config"
    parser_output.mkdir(parents=True, exist_ok=True)
    output_path = parser_output / "morph_signature_external_support.csv"

    corpus = Corpus.load()
    counts = build_external_signature_counts(corpus.api)

    with open(output_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["signature", "count"])
        for signature, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([signature, count])

    print(f"wrote {len(counts)} distinct signatures to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
