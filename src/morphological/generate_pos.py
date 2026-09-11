"""Computes and writes the POS-only skeleton: unigram, bigram, trigram, both scales."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.cli import run_generator
from core.ngram_dataset import NgramDataset, generate_ngram_dataset
from morphological import DATASET_TYPE
from morphological.corpus import Corpus, MorphologicalPsalm
from morphological.vocabulary import SP_VOCABULARY, sp_columns

#: The POS vocabulary is small enough that even the trigram block is stored densely.
DATASET = NgramDataset(
    unit="sp",
    domain=DATASET_TYPE,
    columns_of=sp_columns,
    vocabulary=SP_VOCABULARY,
    constructions={
        "1gram": (1,),
        "1_2gram": (1, 2),
        "1_2_3gram": (1, 2, 3),
        "1gram_psalm": (1,),
        "1_2gram_psalm": (1, 2),
        "1_2_3gram_psalm": (1, 2, 3),
    },
    description="POS-only grammatical skeleton",
)


def generate(
    psalms: list[MorphologicalPsalm], output_root: Path, *, max_workers: int | None = None
) -> list[str]:
    """Writes every not-yet-written POS construction, returns the names written."""
    return generate_ngram_dataset(DATASET, psalms, output_root, max_workers=max_workers)


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus] = Corpus.load,
) -> None:
    """Generates every missing POS dataset."""
    run_generator(__doc__, generate, argv, corpus_factory=corpus_factory)


if __name__ == "__main__":
    main()
