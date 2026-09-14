import json
from pathlib import Path

import pytest

from core.driver import (
    Cell,
    Plan,
    available_resources,
    cell_for,
    cells_for_support,
    plan,
    render_rules,
    run_cell,
)
from core.spec import GeneratorSpec, SupportSpec


def _spec(module: str = "syntactic.generate_typ", **kwargs) -> GeneratorSpec:
    return GeneratorSpec(
        module=module,
        partitions=("domain=syntactic/level=phrase/feature=typ/construction=1gram",),
        **kwargs,
    )


class TestCellFor:
    def test_names_the_cell_after_its_module_and_lists_its_files(self, tmp_path: Path) -> None:
        """A cell's outputs are the parquet parts, its inputs the declared support tables."""
        spec = _spec(support=("phrase_signature_external_support.csv",))
        cell = cell_for(spec, tmp_path / "data", tmp_path / "config")
        assert cell.name == "syntactic.generate_typ"
        assert cell.outputs == (
            tmp_path
            / "data/domain=syntactic/level=phrase/feature=typ/construction=1gram/part-0.parquet",
        )
        assert cell.inputs == (tmp_path / "config/phrase_signature_external_support.csv",)
        assert cell.command_args == [
            "--output-root",
            str(tmp_path / "data"),
            "--config-root",
            str(tmp_path / "config"),
        ]

    def test_semantic_cell_carries_its_model_flag_and_resource(self, tmp_path: Path) -> None:
        """One semantic cell is one model, run through the shared generator with --model."""
        spec = GeneratorSpec(
            module="semantic.generate:berel",
            partitions=("domain=semantic/model=berel/text=consonantal",),
            resource="gpu",
        )
        cell = cell_for(spec, tmp_path, tmp_path / "c")
        assert cell.name == "semantic.generate.berel"
        assert cell.module == "semantic.generate"
        assert cell.command_args == ["--output-root", str(tmp_path), "--model", "berel"]
        assert cell.resource == "gpu"


class TestCellsForSupport:
    def test_support_cells_write_into_the_config_root(self, tmp_path: Path) -> None:
        """A builder cell's outputs are the CSVs it declares, with no inputs."""
        (cell,) = cells_for_support(
            [SupportSpec(module="syntactic.scripts.compute_signature_support", outputs=("a.csv",))],
            tmp_path / "config",
        )
        assert cell.outputs == (tmp_path / "config/a.csv",)
        assert cell.inputs == ()
        assert cell.command_args == ["--config-root", str(tmp_path / "config")]


class TestAvailableResources:
    def test_api_is_available_when_every_provider_key_is_set(self) -> None:
        """Hosted-model cells run only when the key their provider reads is present."""
        env = {"TEHILLIM_OPENROUTER_API_KEY": "k", "TEHILLIM_COHERE_API_KEY": "k"}
        assert available_resources(env, gpu=False) == {None, "api"}

    def test_gpu_is_a_declared_fact_not_a_probe(self) -> None:
        """The run states whether a GPU is present, so a missing one blocks rather than fails."""
        assert available_resources({}, gpu=True) == {None, "gpu"}


class TestPlan:
    def test_partitions_every_cell_into_runnable_and_blocked(self, tmp_path: Path) -> None:
        """Blocked cells are named, never silently dropped from the expected set."""
        specs = [
            _spec(),
            GeneratorSpec(
                module="semantic.generate:kalm-embedding",
                partitions=("domain=semantic/model=k/text=consonantal",),
                resource="gpu",
            ),
        ]
        result = plan(specs, [], tmp_path / "d", tmp_path / "c", available={None})
        assert [c.name for c in result.runnable] == ["syntactic.generate_typ"]
        assert [c.name for c in result.blocked] == ["semantic.generate.kalm-embedding"]
        assert len(result.expected_outputs) == 2


class TestRenderRules:
    def test_emits_one_rule_per_cell_with_provenance_params(self, tmp_path: Path) -> None:
        """Each rule carries its outputs, inputs, resource and a params hash for staleness."""
        cell = Cell(
            name="syntactic.generate_typ",
            module="syntactic.generate_typ",
            inputs=(tmp_path / "c/a.csv",),
            outputs=(tmp_path / "d/x/part-0.parquet",),
            command_args=["--output-root", "d"],
            resource=None,
        )
        text = render_rules([cell], python="py", provenance={"syntactic.generate_typ": "abc"})
        assert "rule cell__syntactic_generate_typ:" in text
        assert "provenance=" in text
        assert "abc" in text
        assert "--cell syntactic.generate_typ" in text
        assert "resources:" not in text

    def test_gpu_and_api_cells_declare_their_resource(self, tmp_path: Path) -> None:
        cell = Cell("semantic.generate.k", "semantic.generate", (), (tmp_path / "p",), [], "gpu")
        text = render_rules([cell], python="py", provenance={"semantic.generate.k": "h"})
        assert "resources:\n        gpu=1" in text


class TestRunCell:
    def test_runs_the_module_main_and_writes_a_manifest_per_output_directory(
        self, tmp_path: Path
    ) -> None:
        """The cell runner is the one place a generator is invoked and provenance recorded."""
        out = tmp_path / "d/domain=x/unit=y/construction=z/part-0.parquet"
        support = tmp_path / "c/s.csv"
        support.parent.mkdir()
        support.write_text("a,1\n")
        calls: list[list[str]] = []

        def fake_main(argv: list[str]) -> None:
            calls.append(argv)
            out.parent.mkdir(parents=True)
            out.write_bytes(b"vectors")

        cell = Cell("m", "m", (support,), (out,), ["--output-root", "d"], None)
        run_cell(cell, main=fake_main, revision=lambda: "rev", code_hash_of=lambda m: "code")
        assert calls == [["--output-root", "d"]]
        manifest = json.loads((out.parent / "_manifest.json").read_text())
        assert manifest["cell"] == "m"
        assert manifest["code_hash"] == "code"
        assert manifest["repository_revision"] == "rev"
        assert list(manifest["inputs"]) == [str(support)]
        assert list(manifest["outputs"]) == [str(out)]

    def test_a_missing_output_after_main_is_an_error(self, tmp_path: Path) -> None:
        """A generator that returns without writing its partition fails the cell."""
        cell = Cell("m", "m", (), (tmp_path / "never/part-0.parquet",), [], None)
        with pytest.raises(FileNotFoundError, match="never"):
            run_cell(cell, main=lambda argv: None, revision=lambda: "r", code_hash_of=lambda m: "c")


class TestRunManifest:
    def test_records_expected_complete_and_blocked_and_fails_on_a_gap(self, tmp_path: Path) -> None:
        """The run manifest is the completeness contract: every expected output, by status."""
        from core.driver import RunIncompleteError, write_run_manifest

        present = tmp_path / "d/a/part-0.parquet"
        present.parent.mkdir(parents=True)
        present.write_bytes(b"x")
        absent = tmp_path / "d/b/part-0.parquet"
        runnable = Cell("ra", "ra", (), (present,), [], None)
        blocked = Cell("rb", "rb", (), (absent,), [], "gpu")
        result = plan([], [], tmp_path / "d", tmp_path / "c", available={None})
        result = Plan(runnable=(runnable,), blocked=(blocked,))
        with pytest.raises(RunIncompleteError, match="rb"):
            write_run_manifest(result, tmp_path / "d", revision=lambda: "rev")
        record = json.loads((tmp_path / "d/_manifest.json").read_text())
        assert record["complete"] == [str(present)]
        assert record["blocked"] == {"rb": "gpu"}
        assert record["missing"] == [str(absent)]
        assert record["expected_cells"] == 2


class TestProvenanceOf:
    def test_carries_code_identity_and_every_present_input_hash(self, tmp_path: Path) -> None:
        """The params string changes when the code or any input's content changes."""
        from core.driver import provenance_of

        support = tmp_path / "s.csv"
        support.write_text("a,1\n")
        cell = Cell("m", "m", (support,), (), [], None)
        before = provenance_of(cell, code_hash_of=lambda m: "code")
        support.write_text("a,2\n")
        after = provenance_of(cell, code_hash_of=lambda m: "code")
        assert before != after
        assert json.loads(before)["code"] == "code"

    def test_the_worker_count_is_not_part_of_a_cells_identity(self) -> None:
        """Running with more workers changes wall clock, never the result."""
        from core.driver import provenance_of

        four = Cell("m", "m", (), (), ["--output", "a.csv", "--workers", "4"], None)
        eight = Cell("m", "m", (), (), ["--output", "a.csv", "--workers", "8"], None)
        assert provenance_of(four, code_hash_of=lambda m: "c") == provenance_of(
            eight, code_hash_of=lambda m: "c"
        )

    def test_changing_a_cell_argument_changes_its_provenance(self) -> None:
        """A rewired input or flag reruns the cell even when code and files are unchanged."""
        from core.driver import provenance_of

        one = Cell("m", "m", (), (), ["--summary-csv", "a.csv"], None)
        other = Cell("m", "m", (), (), ["--summary-csv", "b.csv"], None)
        assert provenance_of(one, code_hash_of=lambda m: "c") != provenance_of(
            other, code_hash_of=lambda m: "c"
        )


class TestMain:
    def test_manifest_subcommand_writes_the_run_record(self, tmp_path: Path) -> None:
        """The final rule calls this to seal a run."""
        from core.driver import main

        out = tmp_path / "d/a/part-0.parquet"
        out.parent.mkdir(parents=True)
        out.write_bytes(b"x")
        result = Plan(runnable=(Cell("a", "a", (), (out,), [], None),), blocked=())
        main(
            ["manifest", "--data-root", str(tmp_path / "d"), "--config-root", str(tmp_path / "c")],
            plan_factory=lambda data_root, config_root, gpu: result,
        )
        assert json.loads((tmp_path / "d/_manifest.json").read_text())["missing"] == []

    def test_run_rejects_an_unplanned_cell(self, tmp_path: Path) -> None:
        """A cell name the plan does not know is a usage error, not a silent no-op."""
        from core.driver import main

        empty = Plan(runnable=(), blocked=())
        with pytest.raises(SystemExit):
            main(
                ["run", "--cell", "nope", "--config-root", str(tmp_path)],
                plan_factory=lambda data_root, config_root, gpu: empty,
            )


class TestRenderRulesRunner:
    def test_a_consumer_repository_names_its_runner_and_thread_count(self, tmp_path: Path) -> None:
        """The benchmark renders rules through its driver module and reserves its worker threads."""
        cell = Cell(
            "genre.lexical.summary",
            "genre.scripts.compare_models",
            (),
            (tmp_path / "s.csv",),
            [],
            None,
        )
        text = render_rules(
            [cell], python="py", provenance={cell.name: "h"}, runner="library.driver", threads=4
        )
        assert "py -m library.driver run --cell genre.lexical.summary" in text
        assert "    threads: 4" in text
