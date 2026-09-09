"""Computes whole-Bible-outside-Psalms clause vocabularies: the frozen label-blind base."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Any

from core.cli import add_config_root_argument
from core.corpus import PSALMS_BOOK_NAME
from core.support import write_external_signature_counts
from syntactic.clause_signature import build_clause_signature
from syntactic.corpus import ClausePsalm, Corpus, clause_corpus

#: Each vocabulary is counted on the node type it will later be applied to.
_VOCABULARIES: tuple[tuple[str, str, Callable[[Any, int], str]], ...] = (
    (
        "clause_atom_typ_external_support.csv",
        "clause_atom",
        lambda api, node: str(api.F.typ.v(node)),
    ),
    (
        "clause_typ_external_support.csv",
        "clause",
        lambda api, node: str(api.F.typ.v(node)),
    ),
    (
        "clause_rela_external_support.csv",
        "clause",
        lambda api, node: str(api.F.rela.v(node)),
    ),
    (
        "clause_signature_external_support.csv",
        "clause",
        lambda api, node: build_clause_signature(
            typ=str(api.F.typ.v(node)), rela=str(api.F.rela.v(node))
        ),
    ),
)


def build_external_counts(
    api: Any, otype: str, token_of: Callable[[Any, int], str]
) -> dict[str, int]:
    """Tallies `token_of` over every `otype` node outside the Psalms book."""
    F, L = api.F, api.L  # noqa: N806
    counts: dict[str, int] = {}
    for book in F.otype.s("book"):
        if F.book.v(book) == PSALMS_BOOK_NAME:
            continue
        for node in L.d(book, otype=otype):
            token = token_of(api, node)
            counts[token] = counts.get(token, 0) + 1
    return counts


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus[ClausePsalm]] = clause_corpus,
) -> None:
    """Writes every clause external-support CSV under the given config root."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_root_argument(parser)
    args = parser.parse_args(argv)
    api = corpus_factory().api
    for filename, otype, token_of in _VOCABULARIES:
        counts = build_external_counts(api, otype, token_of)
        write_external_signature_counts(args.config_root / filename, counts)
        print(f"wrote {len(counts)} distinct {otype} tokens to {filename}", file=sys.stderr)


if __name__ == "__main__":
    main()
