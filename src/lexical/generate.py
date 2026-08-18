"""Computes and writes binary-presence lexical embedding datasets as Parquet."""

from __future__ import annotations

import sys
from pathlib import Path

from lexical.corpus import Corpus, LexicalPsalm
from lexical.export import dataset_path, write_dataset
from lexical.vectorize import binary_presence_vectors
from lexical.vocabulary import VocabularyKey, build_vocabulary

_VOCABULARIES: tuple[tuple[str, VocabularyKey], ...] = (("form", "lex0"), ("lexeme", "lex"))
_WEIGHT = "binary"


def generate(psalms: list[LexicalPsalm], output_root: Path) -> list[str]:
    """Writes every not-yet-written (vocab, weight=binary) dataset, returns the names written."""
    written: list[str] = []
    for vocab_name, key in _VOCABULARIES:
        if dataset_path(output_root, vocab_name, _WEIGHT).exists():
            continue
        print(f"computing lexical vocab={vocab_name} weight={_WEIGHT}...", file=sys.stderr)
        vocabulary = build_vocabulary(psalms, key=key)
        vectors = binary_presence_vectors(psalms, vocabulary, key=key)
        description = (
            f"Binary lexical presence vectors over the {vocab_name} vocabulary "
            f"({key} BHSA feature), dimension {len(vocabulary)}."
        )
        write_dataset(output_root, vocab_name, _WEIGHT, vectors, description)
        written.append(f"{vocab_name}_{_WEIGHT}")
    return written


def main() -> None:
    """Generates every missing lexical dataset for both vocabulary variants."""
    output_root = Path(__file__).resolve().parents[2]
    psalms = Corpus.load().psalms()
    written = generate(psalms, output_root)
    print(f"wrote {len(written)} dataset files", file=sys.stderr)


if __name__ == "__main__":
    main()
