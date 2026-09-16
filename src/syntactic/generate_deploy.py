"""Computes and writes the psalm-scale phrase-signature deployment representation (Phase 5F)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.cli import run_signature_generator
from core.export import path_to_write, write_vectors
from core.partition import BHSA_HALF_VERSE, Partition, family
from core.spec import GeneratorSpec
from core.support import build_signature_vocabulary
from syntactic import DOMAIN, SIGNATURE_FEATURE
from syntactic.corpus import Corpus, PhrasePsalm, phrase_corpus
from syntactic.deploy import signature_deploy_vectors
from syntactic.signature_support import MIN_EXTERNAL_SUPPORT_K

CONSTRUCTION = "posmean"
_SUPPORT_FILE = "phrase_signature_external_support.csv"

SPEC = GeneratorSpec(
    module=__name__,
    partitions=family(
        BHSA_HALF_VERSE, DOMAIN, (CONSTRUCTION,), level="phrase", feature=SIGNATURE_FEATURE
    ),
    support=(_SUPPORT_FILE,),
)


def generate(
    psalms: list[PhrasePsalm], output_root: Path, external_counts: dict[str, int], k: int
) -> list[str]:
    """Writes the phrase_signature posmean dataset if not already present, returns names."""
    partition = Partition(
        BHSA_HALF_VERSE,
        DOMAIN,
        level="phrase",
        feature=SIGNATURE_FEATURE,
        construction=CONSTRUCTION,
    )
    path = path_to_write(output_root, partition)
    if path is None:
        return []
    vocabulary = build_signature_vocabulary(external_counts, k)
    vectors = signature_deploy_vectors(psalms, vocabulary, external_counts, k)
    description = f"Psalm-scale phrase-signature deployment [b;m], construction={CONSTRUCTION}."
    write_vectors(path, vectors, description)
    return [partition.identifier]


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus[PhrasePsalm]] = phrase_corpus,
) -> None:
    """Generates the phrase_signature posmean dataset if missing."""
    run_signature_generator(
        __doc__,
        generate,
        _SUPPORT_FILE,
        MIN_EXTERNAL_SUPPORT_K,
        argv,
        corpus_factory=corpus_factory,
    )


if __name__ == "__main__":
    main()
