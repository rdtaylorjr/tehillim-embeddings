from pathlib import Path

import pytest

from core.spec import GeneratorSpec, discover_specs, partition_dirs, partition_paths


class TestPartitionDirs:
    def test_builds_hive_dirs_in_bhsa_tree_order(self) -> None:
        """domain, then level, then the unit key, then text, then construction."""
        dirs = partition_dirs(
            "syntactic", "feature", "typ", ("1gram", "1gram_psalm"), level="phrase"
        )
        assert dirs == (
            "domain=syntactic/level=phrase/feature=typ/construction=1gram",
            "domain=syntactic/level=phrase/feature=typ/construction=1gram_psalm",
        )

    def test_text_tier_sits_between_unit_and_construction(self) -> None:
        """Lexical surface datasets carry a text tier after the unit."""
        (only,) = partition_dirs("lexical", "unit", "word", ("icf",), text="vocalized")
        assert only == "domain=lexical/unit=word/text=vocalized/construction=icf"

    def test_semantic_layout_has_model_and_text_only(self) -> None:
        """Semantic partitions use model as the unit key and no construction segment."""
        (only,) = partition_dirs("semantic", "model", "berel", (), text="consonantal")
        assert only == "domain=semantic/model=berel/text=consonantal"


class TestPartitionPaths:
    def test_appends_the_parquet_part_under_the_root(self, tmp_path: Path) -> None:
        """Every partition holds exactly one part-0.parquet file."""
        spec = GeneratorSpec(module="m", partitions=("domain=x/unit=y/construction=z",))
        assert partition_paths(spec, tmp_path) == (
            tmp_path / "domain=x/unit=y/construction=z/part-0.parquet",
        )


class TestGeneratorSpec:
    def test_rejects_a_spec_that_writes_nothing(self) -> None:
        """A generator with no partitions is a mistake, never a valid declaration."""
        with pytest.raises(ValueError, match="no partitions"):
            GeneratorSpec(module="m", partitions=())

    def test_rejects_duplicate_partitions(self) -> None:
        """Two rules writing one path would race, so the spec refuses it."""
        with pytest.raises(ValueError, match="duplicate"):
            GeneratorSpec(module="m", partitions=("a", "a"))


class TestDiscoverSpecs:
    def test_finds_every_generator_module_and_each_declares_a_spec(self) -> None:
        """Every generate* module under the family packages exposes SPEC or SPECS."""
        specs = discover_specs()
        modules = {spec.module for spec in specs}
        assert "lexical.generate" in modules
        assert "syntactic.generate_typ" in modules
        assert "semantic.generate" in {m.rsplit(":", 1)[0] for m in modules}

    def test_no_two_specs_claim_one_partition(self) -> None:
        """Partitions are owned by exactly one generator."""
        seen: dict[str, str] = {}
        for spec in discover_specs():
            for partition in spec.partitions:
                assert partition not in seen, (partition, seen.get(partition), spec.module)
                seen[partition] = spec.module

    def test_declared_partitions_match_the_generated_tree(self) -> None:
        """The specs reproduce the tree the hand runs produced, and nothing else."""
        root = Path(__file__).resolve().parents[1] / "data"
        if not root.exists():
            pytest.skip("no generated tree in this checkout")
        on_disk = {str(p.parent.relative_to(root)) for p in root.rglob("part-0.parquet")}
        declared = {p for spec in discover_specs() for p in spec.partitions}
        assert declared == on_disk


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
