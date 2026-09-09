"""Holds every generator script to one definition of the arguments and report they share."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from core.cli import (
    add_config_root_argument,
    add_output_root_argument,
    add_shuffle_arguments,
    report_written,
    run_generator,
    run_signature_generator,
)
from core.shuffle import DEFAULT_N_SHUFFLES


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser()


class TestOutputRoot:
    def test_is_required_because_a_generator_has_nowhere_else_to_write(self) -> None:
        parser = _parser()
        add_output_root_argument(parser)

        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parses_as_a_path(self) -> None:
        parser = _parser()
        add_output_root_argument(parser)

        assert parser.parse_args(["--output-root", "/tmp/out"]).output_root == Path("/tmp/out")


class TestConfigRoot:
    def test_is_required_because_a_signature_vocabulary_needs_its_support_table(self) -> None:
        parser = _parser()
        add_config_root_argument(parser)

        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parses_as_a_path(self) -> None:
        parser = _parser()
        add_config_root_argument(parser)

        assert parser.parse_args(["--config-root", "/tmp/cfg"]).config_root == Path("/tmp/cfg")


class TestShuffleArguments:
    def test_the_draw_count_defaults_to_the_shared_constant(self) -> None:
        parser = _parser()
        add_shuffle_arguments(parser)

        assert parser.parse_args([]).n_shuffles == DEFAULT_N_SHUFFLES

    def test_the_draw_count_can_be_overridden(self) -> None:
        parser = _parser()
        add_shuffle_arguments(parser)

        assert parser.parse_args(["--n-shuffles", "7"]).n_shuffles == 7

    def test_workers_defaults_to_none_so_the_pool_sizes_itself(self) -> None:
        parser = _parser()
        add_shuffle_arguments(parser)

        assert parser.parse_args([]).max_workers is None


class TestReportWritten:
    def test_reports_the_count_on_stderr_which_is_the_scripts_progress_interface(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        report_written(["a", "b", "c"])

        assert "wrote 3 shuffle-control datasets" in capsys.readouterr().err

    def test_writes_nothing_to_stdout_which_carries_no_progress(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        report_written(["a"])

        assert capsys.readouterr().out == ""


class _FakeCorpus:
    """A corpus stand-in whose psalms the driver passes straight through to the generator."""

    def __init__(self, psalms: list[str]) -> None:
        self._psalms = psalms

    def psalms(self) -> list[str]:
        return self._psalms


class TestRunGenerator:
    def test_passes_the_corpus_psalms_and_output_root_to_the_generator(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        seen: list[tuple[list[str], Path]] = []

        def generate(psalms: list[str], output_root: Path) -> list[str]:
            seen.append((psalms, output_root))
            return []

        run_generator(
            "doc",
            generate,
            ["--output-root", str(tmp_path)],
            corpus_factory=lambda: _FakeCorpus(["ps1", "ps2"]),
        )
        capsys.readouterr()

        assert seen == [(["ps1", "ps2"], tmp_path)]

    def test_reports_how_many_datasets_the_generator_wrote(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run_generator(
            "doc",
            lambda _psalms, _root: ["a", "b"],
            ["--output-root", str(tmp_path)],
            corpus_factory=lambda: _FakeCorpus([]),
        )

        assert "wrote 2 dataset files" in capsys.readouterr().err

    def test_requires_an_output_root_because_a_generator_has_nowhere_else_to_write(
        self,
    ) -> None:
        with pytest.raises(SystemExit):
            run_generator(
                "doc",
                lambda _psalms, _root: [],
                [],
                corpus_factory=lambda: _FakeCorpus([]),
            )

    def test_does_not_load_the_corpus_before_the_arguments_parse(self) -> None:
        """A bad command line must fail before the corpus is loaded, which is the slow step."""

        def exploding_factory() -> _FakeCorpus:
            raise AssertionError("the corpus must not load when the arguments are invalid")

        with pytest.raises(SystemExit):
            run_generator("doc", lambda _psalms, _root: [], [], corpus_factory=exploding_factory)


class TestRunSignatureGenerator:
    def test_reads_the_support_table_the_named_file_holds(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "sig_external_support.csv").write_text("signature,count\nA,7\n")
        seen: list[dict[str, int]] = []

        def generate(
            _psalms: list[str], _root: Path, external_counts: dict[str, int], _k: int
        ) -> list[str]:
            seen.append(external_counts)
            return []

        run_signature_generator(
            "doc",
            generate,
            "sig_external_support.csv",
            3,
            ["--output-root", str(tmp_path), "--config-root", str(tmp_path)],
            corpus_factory=lambda: _FakeCorpus([]),
        )
        capsys.readouterr()

        assert seen == [{"A": 7}]

    def test_passes_the_support_threshold_through_unchanged(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "sig_external_support.csv").write_text("signature,count\nA,7\n")
        seen: list[int] = []

        def generate(_psalms: list[str], _root: Path, _counts: dict[str, int], k: int) -> list[str]:
            seen.append(k)
            return []

        run_signature_generator(
            "doc",
            generate,
            "sig_external_support.csv",
            5,
            ["--output-root", str(tmp_path), "--config-root", str(tmp_path)],
            corpus_factory=lambda: _FakeCorpus([]),
        )
        capsys.readouterr()

        assert seen == [5]
