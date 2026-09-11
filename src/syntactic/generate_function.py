"""Computes and writes the phrase-function skeleton: 1gram, bigram, trigram, both scales."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.cli import run_generator
from core.ngram_dataset import NgramDataset, generate_ngram_dataset
from syntactic import DATASET_TYPE
from syntactic.corpus import Corpus, PhrasePsalm, phrase_corpus
from syntactic.vocabulary import FUNCTION_VOCABULARY, function_columns

#: The trigram block is overwhelmingly zero at this dimension, so it is stored sparsely.
DATASET = NgramDataset(
    unit="function",
    domain=DATASET_TYPE,
    level="phrase",
    columns_of=function_columns,
    vocabulary=FUNCTION_VOCABULARY,
    constructions={
        "1gram": (1,),
        "1_2gram": (1, 2),
        "1_2_3gram": (1, 2, 3),
        "1gram_psalm": (1,),
        "1_2gram_psalm": (1, 2),
        "1_2_3gram_psalm": (1, 2, 3),
    },
    sparse=frozenset({"1_2_3gram", "1_2_3gram_psalm"}),
    description="Phrase-function-only skeleton",
)


def generate(
    psalms: list[PhrasePsalm], output_root: Path, *, max_workers: int | None = None
) -> list[str]:
    """Writes every not-yet-written phrase-function construction, returns the names written."""
    return generate_ngram_dataset(DATASET, psalms, output_root, max_workers=max_workers)


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus[PhrasePsalm]] = phrase_corpus,
) -> None:
    """Generates every missing phrase-function dataset."""
    run_generator(__doc__, generate, argv, corpus_factory=corpus_factory)


if __name__ == "__main__":
    main()
