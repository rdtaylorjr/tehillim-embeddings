"""Computes and writes lexical datasets as Parquet: 19 weightings each for homograph and lexeme."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core.cli import add_output_root_argument, report_generated
from core.columns import PsalmColumns
from core.export import path_to_write, write_vectors
from core.parallel import map_items
from lexical.constructions import FULL_WEIGHTS, vectors_for_weight
from lexical.corpus import Corpus, LexicalPsalm
from lexical.frequency import icf_weights as compute_icf_weights
from lexical.frequency import lex0_token_frequencies, lex_token_frequencies, total_token_count
from lexical.vocabulary import VocabularyKey, build_vocabulary, columns_for_key

_VOCAB_NAMES: dict[VocabularyKey, str] = {"lex0": "homograph", "lex": "lexeme"}


@dataclass(frozen=True, slots=True)
class _VocabularyContext:
    """One vocabulary's inputs, pickled once per worker rather than once per construction."""

    key: VocabularyKey
    vocab_name: str
    output_root: Path
    vocabulary: tuple[str, ...]
    columns: list[PsalmColumns]
    icf_weights: dict[str, float]


def write_construction(context: _VocabularyContext, weight: str) -> str | None:
    """Writes one construction's dataset, or returns None when it is already written."""
    path = path_to_write(context.output_root, context.vocab_name, weight, unit_key="unit")
    if path is None:
        return None
    vectors = vectors_for_weight(context.columns, context.vocabulary, weight, context.icf_weights)
    description = (
        f"Lexical vectors over the {context.vocab_name} unit (BHSA {context.key} feature), "
        f"construction={weight}, dimension {len(context.vocabulary)}."
    )
    write_vectors(path, vectors, description)
    return f"{context.vocab_name}_{weight}"


def generate(
    psalms: list[LexicalPsalm],
    output_root: Path,
    icf_weights_by_key: dict[VocabularyKey, dict[str, float]],
    *,
    max_workers: int | None = None,
) -> list[str]:
    """Writes every not-yet-written (vocab, weight) dataset, returns the names written."""
    written: list[str] = []
    for key in ("lex0", "lex"):
        context = _VocabularyContext(
            key=key,
            vocab_name=_VOCAB_NAMES[key],
            output_root=output_root,
            #: Built once per vocabulary rather than once per construction.
            vocabulary=build_vocabulary(psalms, key=key),
            columns=columns_for_key(psalms, key),
            icf_weights=icf_weights_by_key[key],
        )
        names = map_items(write_construction, context, FULL_WEIGHTS, max_workers=max_workers)
        written.extend(name for name in names if name is not None)
    return written


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus] = Corpus.load,
) -> None:
    """Generates every missing lexical dataset: 19 weightings each for homograph and lexeme."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_output_root_argument(parser)
    args = parser.parse_args(argv)
    corpus = corpus_factory()
    psalms = corpus.psalms()
    total_tokens = total_token_count(corpus.api)
    icf_weights_by_key: dict[VocabularyKey, dict[str, float]] = {
        "lex0": compute_icf_weights(lex0_token_frequencies(corpus.api), total_tokens),
        "lex": compute_icf_weights(lex_token_frequencies(corpus.api), total_tokens),
    }
    written = generate(psalms, args.output_root, icf_weights_by_key)
    report_generated(written)


if __name__ == "__main__":
    main()
