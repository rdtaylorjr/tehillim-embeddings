"""Shared driver for the phrase skeleton generators, each a per-node and psalm pair."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from core.export import dataset_path, write_vectors
from syntax.corpus import Corpus, PhrasePsalm

_DATASET_TYPE = "syntax"

Builder = Callable[[list[PhrasePsalm]], dict[int, np.ndarray]]


def generate_skeleton(
    psalms: list[PhrasePsalm],
    output_root: Path,
    unit: str,
    constructions: Sequence[tuple[str, Builder]],
    description: str,
) -> list[str]:
    """Writes each not-yet-written `<unit>` construction, returns the qualified names written."""
    written: list[str] = []
    for construction, builder in constructions:
        path = dataset_path(
            output_root,
            unit,
            construction,
            domain=_DATASET_TYPE,
            unit_key="feature",
            level="phrase",
        )
        if path.exists():
            continue
        print(f"computing syntax feature={unit} construction={construction}...", file=sys.stderr)
        write_vectors(path, builder(psalms), f"{description}, construction={construction}.")
        written.append(f"{unit}_{construction}")
    return written


def run(
    doc: str | None,
    generate: Callable[[list[PhrasePsalm], Path], list[str]],
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus] = Corpus.load,
) -> None:
    """Parses `--output-root`, generates, and reports how many datasets were written."""
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    written = generate(corpus_factory().psalms(), args.output_root)
    print(f"wrote {len(written)} dataset files", file=sys.stderr)
