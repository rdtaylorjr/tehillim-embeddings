"""Computes whole-Bible-outside-Psalms phrase-signature frequencies: the frozen label-blind base."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from phrase.corpus import Corpus
from phrase.signature import build_phrase_signature

_PSALMS_BOOK_NAME = "Psalmi"


def build_external_signature_counts(api: Any) -> dict[str, int]:
    """Tallies `build_phrase_signature(...)` over every phrase atom outside the Psalms book."""
    F, L = api.F, api.L  # noqa: N806
    counts: dict[str, int] = {}
    for book in F.otype.s("book"):
        if F.book.v(book) == _PSALMS_BOOK_NAME:
            continue
        for atom in L.d(book, otype="phrase_atom"):
            mother = L.u(atom, otype="phrase")
            signature = build_phrase_signature(typ=F.typ.v(atom), function=F.function.v(mother[0]))
            counts[signature] = counts.get(signature, 0) + 1
    return counts


def main() -> None:
    """Writes `config/phrase_signature_external_support.csv` under the given output root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    output_root = parser.parse_args().output_root
    config_dir = output_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    output_path = config_dir / "phrase_signature_external_support.csv"

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
