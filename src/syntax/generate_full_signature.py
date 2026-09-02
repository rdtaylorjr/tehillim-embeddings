"""Computes and writes the full (typ:function:det) signature inventory: H5.8's S+det test."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.cli import run_signature_generator
from core.export import path_to_write, write_vectors
from core.support import build_signature_vocabulary
from syntax import DATASET_TYPE
from syntax.corpus import Corpus, PhrasePsalm
from syntax.full_signature_vectorize import (
    phrase_full_signature_psalm_vectors,
    phrase_full_signature_vectors,
)
from syntax.signature_support import MIN_EXTERNAL_SUPPORT_K_FULL

_UNIT = "full_signature"


def generate(
    psalms: list[PhrasePsalm], output_root: Path, external_counts: dict[str, int], k: int
) -> list[str]:
    """Writes both not-yet-written phrase_full_signature constructions, returns the names."""
    vocabulary = build_signature_vocabulary(external_counts, k)

    written: list[str] = []
    for construction, builder in (
        ("inventory", phrase_full_signature_vectors),
        ("inventory_psalm", phrase_full_signature_psalm_vectors),
    ):
        path = path_to_write(
            output_root,
            _UNIT,
            construction,
            domain=DATASET_TYPE,
            unit_key="feature",
            level="phrase",
        )
        if path is None:
            continue
        vectors = builder(psalms, vocabulary, external_counts, k)
        description = (
            f"Full typ:function:det signature histogram (RARE-collapsed, k={k}), "
            f"construction={construction}."
        )
        write_vectors(path, vectors, description)
        written.append(f"{_UNIT}_{construction}")
    return written


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus] = Corpus.load,
) -> None:
    """Generates every missing phrase_full_signature dataset."""
    run_signature_generator(
        __doc__,
        generate,
        "phrase_full_signature_external_support.csv",
        MIN_EXTERNAL_SUPPORT_K_FULL,
        argv,
        corpus_factory=corpus_factory,
    )


if __name__ == "__main__":
    main()
