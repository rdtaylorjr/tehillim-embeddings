"""Computes and writes the morph_suffix family: suffix n-grams, and host-signature-plus-suffix."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

from core.cli import run_signature_generator
from core.dataset_family import Construction, generate_family
from core.ngram_dataset import NgramDataset, generate_ngram_dataset
from core.partition import BHSA_HALF_VERSE, family
from core.spec import GeneratorSpec
from core.support import build_signature_vocabulary
from morphological import DOMAIN, SUFFIX_FEATURE
from morphological.corpus import Corpus, MorphologicalPsalm
from morphological.signature_support import MIN_EXTERNAL_SUPPORT_K
from morphological.suffix import (
    SUFFIX_VOCABULARY,
    host_plus_suffix_psalm_vectors,
    host_plus_suffix_vectors,
    psalm_suffix_signatures,
)

_DESCRIPTION = "Pronominal-suffix representation"
_HOST_BUILDERS = (
    ("host_plus_suffix", host_plus_suffix_vectors),
    ("host_plus_suffix_psalm", host_plus_suffix_psalm_vectors),
)
SUPPORT_FILENAME = "morph_signature_external_support.csv"

DATASET: NgramDataset[MorphologicalPsalm] = NgramDataset(
    feature=SUFFIX_FEATURE,
    domain=DOMAIN,
    columns_of=psalm_suffix_signatures,
    vocabulary=SUFFIX_VOCABULARY,
    constructions={
        "1gram": (1,),
        "1_2gram": (1, 2),
        "1_2_3gram": (1, 2, 3),
        "1gram_psalm": (1,),
        "1_2gram_psalm": (1, 2),
        "1_2_3gram_psalm": (1, 2, 3),
    },
    sparse=frozenset({"1_2_3gram", "1_2_3gram_psalm"}),
    description=_DESCRIPTION,
)

SPEC = GeneratorSpec(
    module=__name__,
    partitions=family(
        BHSA_HALF_VERSE,
        DOMAIN,
        (*DATASET.constructions, *(name for name, _ in _HOST_BUILDERS)),
        feature=SUFFIX_FEATURE,
    ),
    support=(SUPPORT_FILENAME,),
)


def _host_constructions(
    external_counts: dict[str, int], k: int
) -> list[tuple[str, Construction[MorphologicalPsalm]]]:
    """The two constructions that bundle the RARE-collapsed host signature with its suffix."""
    vocabulary = build_signature_vocabulary(external_counts, k)
    return [
        (
            name,
            Construction(
                build=partial(
                    builder, signature_vocabulary=vocabulary, external_counts=external_counts, k=k
                ),
                description=f"{_DESCRIPTION}, construction={name}.",
            ),
        )
        for name, builder in _HOST_BUILDERS
    ]


def generate(
    psalms: list[MorphologicalPsalm],
    output_root: Path,
    external_counts: dict[str, int],
    k: int,
    *,
    max_workers: int | None = None,
) -> list[str]:
    """Writes every not-yet-written morph_suffix construction, returns the names written."""
    return generate_ngram_dataset(
        DATASET, psalms, output_root, max_workers=max_workers
    ) + generate_family(
        psalms,
        output_root,
        _host_constructions(external_counts, k),
        feature=SUFFIX_FEATURE,
        domain=DOMAIN,
        max_workers=max_workers,
    )


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus] = Corpus.load,
) -> None:
    """Generates every missing morph_suffix dataset."""
    run_signature_generator(
        __doc__,
        generate,
        SUPPORT_FILENAME,
        MIN_EXTERNAL_SUPPORT_K,
        argv,
        corpus_factory=corpus_factory,
    )


if __name__ == "__main__":
    main()
