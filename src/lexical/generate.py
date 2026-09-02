"""Computes and writes lexical datasets as Parquet: 19 weightings each for homograph and lexeme."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from core.export import dataset_path, write_dataset
from lexical.constructions import FULL_WEIGHTS, vectors_for_weight
from lexical.corpus import Corpus, LexicalPsalm
from lexical.frequency import icf_weights as compute_icf_weights
from lexical.frequency import lex0_token_frequencies, lex_token_frequencies, total_token_count
from lexical.vocabulary import VocabularyKey, build_vocabulary, columns_for_key

_VOCAB_NAMES: dict[VocabularyKey, str] = {"lex0": "homograph", "lex": "lexeme"}


def generate(
    psalms: list[LexicalPsalm],
    output_root: Path,
    icf_weights_by_key: dict[VocabularyKey, dict[str, float]],
) -> list[str]:
    """Writes every not-yet-written (vocab, weight) dataset, returns the names written."""
    written: list[str] = []
    for key in ("lex0", "lex"):
        vocab_name = _VOCAB_NAMES[key]
        #: Built once per vocabulary rather than once per construction.
        vocabulary = build_vocabulary(psalms, key=key)
        columns = columns_for_key(psalms, key)
        for weight in FULL_WEIGHTS:
            if dataset_path(output_root, vocab_name, weight, unit_key="unit").exists():
                continue
            print(f"computing lexical unit={vocab_name} construction={weight}...", file=sys.stderr)
            vectors = vectors_for_weight(columns, vocabulary, weight, icf_weights_by_key[key])
            description = (
                f"Lexical vectors over the {vocab_name} unit (BHSA {key} feature), "
                f"construction={weight}, dimension {len(vocabulary)}."
            )
            write_dataset(output_root, vocab_name, weight, vectors, description, unit_key="unit")
            written.append(f"{vocab_name}_{weight}")
    return written


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus] = Corpus.load,
) -> None:
    """Generates every missing lexical dataset: 19 weightings each for homograph and lexeme."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    corpus = corpus_factory()
    psalms = corpus.psalms()
    total_tokens = total_token_count(corpus.api)
    icf_weights_by_key: dict[VocabularyKey, dict[str, float]] = {
        "lex0": compute_icf_weights(lex0_token_frequencies(corpus.api), total_tokens),
        "lex": compute_icf_weights(lex_token_frequencies(corpus.api), total_tokens),
    }
    written = generate(psalms, args.output_root, icf_weights_by_key)
    print(f"wrote {len(written)} dataset files", file=sys.stderr)


if __name__ == "__main__":
    main()
