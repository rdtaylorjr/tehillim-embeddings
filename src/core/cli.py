"""One definition of the arguments and the completion report the generator scripts share."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from core.shuffle import DEFAULT_N_SHUFFLES
from core.support import load_external_signature_counts, write_external_signature_counts

if TYPE_CHECKING:
    from collections.abc import Callable


def add_output_root_argument(parser: argparse.ArgumentParser) -> None:
    """The Hive-partitioned root every generated dataset is written under."""
    parser.add_argument("--output-root", type=Path, required=True)


def add_config_root_argument(parser: argparse.ArgumentParser) -> None:
    """The directory holding the external support tables a signature vocabulary is built from."""
    parser.add_argument("--config-root", type=Path, required=True)


def add_shuffle_arguments(parser: argparse.ArgumentParser) -> None:
    """How many seeded draws a shuffle-null control writes, and across how many worker processes."""
    parser.add_argument("--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES)
    parser.add_argument("--max-workers", type=int, default=None)


def report_written(written: list[str]) -> None:
    """Reports the batch's size on stderr, which is these scripts' progress interface."""
    print(f"wrote {len(written)} shuffle-control datasets", file=sys.stderr)


class PsalmCorpus[PsalmT](Protocol):
    """The one thing a generator needs from a corpus: the psalms it will vectorize."""

    def psalms(self) -> list[PsalmT]:
        """The psalms a generator will vectorize."""
        ...


def _generator_parser(doc: str | None, *, with_config_root: bool) -> argparse.ArgumentParser:
    """The command line every dataset generator shares, plus the support table when one is read."""
    parser = argparse.ArgumentParser(description=doc)
    add_output_root_argument(parser)
    if with_config_root:
        add_config_root_argument(parser)
    return parser


def report_generated(written: list[str]) -> None:
    """Reports the batch's size on stderr, which is these scripts' progress interface."""
    print(f"wrote {len(written)} dataset files", file=sys.stderr)


def run_generator[PsalmT](
    doc: str | None,
    generate: Callable[[list[PsalmT], Path], list[str]],
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], PsalmCorpus[PsalmT]],
) -> None:
    """Parses the shared command line, writes every missing dataset, and reports the count."""
    args = _generator_parser(doc, with_config_root=False).parse_args(argv)
    written = generate(corpus_factory().psalms(), args.output_root)
    report_generated(written)


def run_signature_generator[PsalmT](
    doc: str | None,
    generate: Callable[[list[PsalmT], Path, dict[str, int], int], list[str]],
    support_filename: str,
    k: int,
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], PsalmCorpus[PsalmT]],
) -> None:
    """Adds the external support table a signature vocabulary is built from to run_generator."""
    args = _generator_parser(doc, with_config_root=True).parse_args(argv)
    external_counts = load_external_signature_counts(args.config_root / support_filename)
    written = generate(corpus_factory().psalms(), args.output_root, external_counts, k)
    report_generated(written)


class ApiCorpus(Protocol):
    """The one thing a support builder needs from a corpus: the Text-Fabric api it counts over."""

    @property
    def api(self) -> Any:
        """The loaded Text-Fabric api."""
        ...


def run_support_builder(
    doc: str | None,
    build_counts: Callable[[Any], dict[str, int]],
    output_filename: str,
    argv: list[str] | None = None,
    *,
    corpus_factory: Callable[[], ApiCorpus],
) -> None:
    """Counts a signature's whole-Bible support outside Psalms and writes it under --config-root."""
    parser = argparse.ArgumentParser(description=doc)
    add_config_root_argument(parser)
    args = parser.parse_args(argv)
    output_path = args.config_root / output_filename
    counts = build_counts(corpus_factory().api)
    write_external_signature_counts(output_path, counts)
    print(f"wrote {len(counts)} distinct signatures to {output_path}", file=sys.stderr)
