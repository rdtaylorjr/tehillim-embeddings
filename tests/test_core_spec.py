from pathlib import Path

import pytest

from core.driver import plan
from core.partition import BHSA_HALF_VERSE, Partition
from core.spec import GeneratorSpec, discover_specs, partition_paths

TYP_1GRAM = Partition(
    BHSA_HALF_VERSE, "syntactic", level="phrase", feature="typ", construction="1gram"
)


class TestPartitionPaths:
    def test_appends_the_parquet_part_under_the_root(self, tmp_path: Path) -> None:
        """Every partition holds exactly one part-0.parquet file."""
        spec = GeneratorSpec(module="m", partitions=(TYP_1GRAM,))
        assert partition_paths(spec, tmp_path) == (
            tmp_path
            / "corpus=bhsa/unit=half_verse/domain=syntactic/level=phrase/feature=typ"
            / "construction=1gram/part-0.parquet",
        )


class TestGeneratorSpec:
    def test_rejects_a_spec_that_writes_nothing(self) -> None:
        """A generator with no partitions is a mistake, never a valid declaration."""
        with pytest.raises(ValueError, match="no partitions"):
            GeneratorSpec(module="m", partitions=())

    def test_rejects_duplicate_partitions(self) -> None:
        """Two rules writing one path would race, so the spec refuses it."""
        with pytest.raises(ValueError, match="duplicate"):
            GeneratorSpec(module="m", partitions=(TYP_1GRAM, TYP_1GRAM))

    def test_a_variant_and_its_args_come_together(self) -> None:
        """A variant with no args could not be selected, args with no variant could not be named."""
        with pytest.raises(ValueError, match="together"):
            GeneratorSpec(module="m", partitions=(TYP_1GRAM,), variant="v")
        with pytest.raises(ValueError, match="together"):
            GeneratorSpec(module="m", partitions=(TYP_1GRAM,), args=("--x",))
        spec = GeneratorSpec(module="m", partitions=(TYP_1GRAM,), variant="v", args=("--x",))
        assert spec.name == "m.v"
        assert GeneratorSpec(module="m", partitions=(TYP_1GRAM,)).name == "m"


class TestDiscoverSpecs:
    def test_finds_every_generator_module_and_each_declares_a_spec(self) -> None:
        """Every generate* module under the family packages exposes SPEC or SPECS."""
        specs = discover_specs()
        modules = {spec.module for spec in specs}
        assert "lexical.generate" in modules
        assert "syntactic.generate_typ" in modules
        assert "semantic.generate" in modules

    def test_no_two_specs_claim_one_partition(self) -> None:
        """Partitions are owned by exactly one generator."""
        seen: dict[Partition, str] = {}
        for spec in discover_specs():
            for partition in spec.partitions:
                assert partition not in seen, (partition, seen.get(partition), spec.module)
                seen[partition] = spec.module

    def test_declared_partitions_match_the_generated_tree(self) -> None:
        """Every file on disk is declared, and every runnable cell has written its files."""
        root = Path(__file__).resolve().parents[1] / "data"
        if not root.exists():
            pytest.skip("no generated tree in this checkout")
        on_disk = {Partition.parse(p.relative_to(root)) for p in root.rglob("part-0.parquet")}
        declared = {p for spec in discover_specs() for p in spec.partitions}
        assert on_disk <= declared
        runnable = plan(discover_specs(), [], root, root, available={None}).runnable
        owed = {p for cell in runnable for p in cell.outputs}
        assert all(p.exists() for p in owed), sorted(str(p) for p in owed if not p.exists())


class TestSupportPaths:
    def test_resolves_each_declared_table_under_the_config_root(self, tmp_path: Path) -> None:
        """A generator that reads two tables gets two paths, in declaration order."""
        from core.spec import support_paths

        spec = GeneratorSpec(module="m", partitions=("a",), support=("x.csv", "y.csv"))
        assert support_paths(spec, tmp_path) == (tmp_path / "x.csv", tmp_path / "y.csv")

    def test_a_generator_without_tables_resolves_to_nothing(self, tmp_path: Path) -> None:
        """Most generators read only BHSA."""
        from core.spec import support_paths

        assert support_paths(GeneratorSpec(module="m", partitions=("a",)), tmp_path) == ()


class TestDiscoverSupportSpecs:
    def test_every_support_builder_declares_the_tables_it_writes(self) -> None:
        """The config CSVs are produced by exactly the declared builders, one owner each."""
        from core.spec import discover_support_specs

        specs = discover_support_specs()
        modules = {spec.module for spec in specs}
        assert "syntactic.scripts.compute_clause_support" in modules
        outputs = [o for spec in specs for o in spec.outputs]
        assert len(outputs) == len(set(outputs))

    def test_declared_tables_match_the_committed_config(self) -> None:
        """The builders account for every table in config and for nothing else."""
        from core.spec import discover_support_specs

        config = Path(__file__).resolve().parents[1] / "config"
        declared = {o for spec in discover_support_specs() for o in spec.outputs}
        assert declared == {p.name for p in config.glob("*.csv")}

    def test_every_generator_support_table_has_a_builder(self) -> None:
        """No generator reads a table nothing produces."""
        from core.spec import discover_support_specs

        produced = {o for spec in discover_support_specs() for o in spec.outputs}
        needed = {t for spec in discover_specs() for t in spec.support}
        assert needed <= produced
