"""Computes and writes the phrase_full_signature family: 1gram, 1_2gram, and 1_2_3gram."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.cli import run_signature_generator
from core.supported_dataset import SupportedDataset, generate_supported_dataset
from syntactic import DATASET_TYPE
from syntactic.corpus import Corpus, PhrasePsalm, phrase_corpus
from syntactic.full_signature_vectorize import DENSE_BUILDERS, SPARSE_BUILDERS
from syntactic.signature_support import MIN_EXTERNAL_SUPPORT_K_FULL

_SUPPORT_FILE = "phrase_full_signature_external_support.csv"

DATASET: SupportedDataset[PhrasePsalm] = SupportedDataset(
    unit="full_signature",
    domain=DATASET_TYPE,
    level="phrase",
    dense=DENSE_BUILDERS,
    sparse=SPARSE_BUILDERS,
    description="Full typ:function:det signature histogram (RARE-collapsed, k={k})",
)


def generate(
    psalms: list[PhrasePsalm],
    output_root: Path,
    external_counts: dict[str, int],
    k: int,
    *,
    max_workers: int | None = None,
) -> list[str]:
    """Writes every not-yet-written phrase_full_signature construction, returns the names."""
    return generate_supported_dataset(
        DATASET, psalms, output_root, external_counts, k, max_workers=max_workers
    )


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus[PhrasePsalm]] = phrase_corpus,
) -> None:
    """Generates every missing phrase_full_signature dataset."""
    run_signature_generator(
        __doc__,
        generate,
        _SUPPORT_FILE,
        MIN_EXTERNAL_SUPPORT_K_FULL,
        argv,
        corpus_factory=corpus_factory,
    )


if __name__ == "__main__":
    main()
