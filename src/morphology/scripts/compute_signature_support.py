"""Computes whole-Bible-outside-Psalms signature frequencies: the frozen, label-blind support."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.cli import run_support_builder
from core.corpus import PSALMS_BOOK_NAME
from morphology.corpus import Corpus
from morphology.signature import build_signature


def build_external_signature_counts(api: Any) -> dict[str, int]:
    """Tallies `build_signature(...)` over every word outside the Psalms book."""
    F, L = api.F, api.L  # noqa: N806
    counts: dict[str, int] = {}
    for book in F.otype.s("book"):
        if F.book.v(book) == PSALMS_BOOK_NAME:
            continue
        for word in L.d(book, otype="word"):
            signature = build_signature(
                sp=F.sp.v(word),
                gn=F.gn.v(word),
                nu=F.nu.v(word),
                ps=F.ps.v(word),
                st=F.st.v(word),
                vs=F.vs.v(word),
                vt=F.vt.v(word),
            )
            counts[signature] = counts.get(signature, 0) + 1
    return counts


def main(
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], Corpus] = Corpus.load,
) -> None:
    """Writes `morph_signature_external_support.csv` under the given config root."""
    run_support_builder(
        __doc__,
        build_external_signature_counts,
        "morph_signature_external_support.csv",
        argv,
        corpus_factory=corpus_factory,
    )


if __name__ == "__main__":
    main()
