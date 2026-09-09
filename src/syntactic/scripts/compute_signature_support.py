"""Computes whole-Bible-outside-Psalms phrase-signature frequencies: the frozen label-blind base."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.cli import run_support_builder
from core.corpus import PSALMS_BOOK_NAME
from syntactic.corpus import Corpus
from syntactic.signature import build_phrase_signature


def build_external_signature_counts(api: Any) -> dict[str, int]:
    """Tallies `build_phrase_signature(...)` over every phrase atom outside the Psalms book."""
    F, L = api.F, api.L  # noqa: N806
    counts: dict[str, int] = {}
    for book in F.otype.s("book"):
        if F.book.v(book) == PSALMS_BOOK_NAME:
            continue
        for atom in L.d(book, otype="phrase_atom"):
            mother = L.u(atom, otype="phrase")
            signature = build_phrase_signature(typ=F.typ.v(atom), function=F.function.v(mother[0]))
            counts[signature] = counts.get(signature, 0) + 1
    return counts


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus] = Corpus.load,
) -> None:
    """Writes `phrase_signature_external_support.csv` under the given config root."""
    run_support_builder(
        __doc__,
        build_external_signature_counts,
        "phrase_signature_external_support.csv",
        argv,
        corpus_factory=corpus_factory,
    )


if __name__ == "__main__":
    main()
