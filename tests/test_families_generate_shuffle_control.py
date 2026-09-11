"""One generator writes every family, so what it writes must equal what the registry builds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from core.export import dataset_path
from core.shuffle import DEFAULT_N_SHUFFLES
from families.scripts.generate_shuffle_control import (
    SeedContext,
    build_parser,
    generate,
    main,
    write_seed,
)
from families.shuffle import FAMILIES, Draws, dataset_source, dataset_target, draw
from morphological import DATASET_TYPE as MORPHOLOGICAL_DOMAIN
from morphological import SUFFIX_UNIT
from morphological.generate_deploy import CONSTRUCTION as MORPHOLOGICAL_CONSTRUCTION
from syntactic import DATASET_TYPE as SYNTACTIC_DOMAIN
from syntactic import SIGNATURE_UNIT
from syntactic.generate_deploy import CONSTRUCTION as SYNTACTIC_CONSTRUCTION

EMBEDDINGS_DATA = Path(__file__).resolve().parents[1] / "data"

DENSE_KEY = "lexical/homograph/icf_position4"
SPARSE_KEY = "syntactic/phrase/typ/1_2_3gram"
NODES = (10, 11)
WIDTH = 3


def _permutation(psalms: list[int], seed: int) -> dict[int, np.ndarray]:
    """One index per node, moved by the seed, so a draw changes when its seed does."""
    return {node: np.array([(node + seed) % WIDTH]) for node in psalms}


def _dense(psalms: list[int], order: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    return {node: np.array([float(order[node][0]), 1.0], dtype="<f4") for node in order}


def _sparse(
    psalms: list[int], order: dict[int, np.ndarray]
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return {
        node: (np.array([int(order[node][0])], dtype=np.int32), np.array([1.0], dtype="<f4"))
        for node in order
    }


def _draws(key: str, build: object, sparse_width: int | None) -> Draws:
    return Draws(
        key=key, psalms=NODES, permute=_permutation, build=build, sparse_width=sparse_width
    )


def _written(path: Path) -> dict[int, object]:
    table = pq.read_table(path)
    nodes = [int(node) for node in table["node_id"].to_pylist()]
    if "vector" in table.schema.names:
        column = table["vector"].combine_chunks()
        flat = column.values.to_numpy(zero_copy_only=False).astype("<f4", copy=False)
        matrix = flat.reshape(len(nodes), column.type.list_size)
        return {node: matrix[row] for row, node in enumerate(nodes)}
    indices, values = table["indices"].to_pylist(), table["values"].to_pylist()
    return {
        node: (
            np.asarray(indices[row], dtype=np.int32),
            np.asarray(values[row], dtype="<f4"),
        )
        for row, node in enumerate(nodes)
    }


class TestDatasetTarget:
    def test_names_the_partition_each_domains_key_describes(self, tmp_path: Path) -> None:
        lexical = dataset_target("lexical/homograph/icf_position4", tmp_path, 7)
        morphological = dataset_target("morphological/sp/1_2gram", tmp_path, 7)
        syntactic = dataset_target("syntactic/phrase/typ/1_2gram", tmp_path, 7)

        assert lexical.relative_to(tmp_path).parts == (
            "domain=lexical",
            "unit=homograph",
            "construction=icf_position4_shuffle0007",
            "part-0.parquet",
        )
        assert morphological.relative_to(tmp_path).parts == (
            "domain=morphological",
            "feature=sp",
            "construction=1_2gram_shuffle0007",
            "part-0.parquet",
        )
        assert syntactic.relative_to(tmp_path).parts == (
            "domain=syntactic",
            "level=phrase",
            "feature=typ",
            "construction=1_2gram_shuffle0007",
            "part-0.parquet",
        )


class TestWriteSeed:
    def test_a_written_dense_draw_reads_back_as_the_registry_built_it(self, tmp_path: Path) -> None:
        draws = _draws(DENSE_KEY, _dense, None)

        write_seed(SeedContext(draws=draws, output_root=tmp_path), 3)

        written = _written(dataset_target(DENSE_KEY, tmp_path, 3))
        built = draw(draws, 3)
        assert sorted(written) == sorted(built)
        assert all(np.array_equal(written[node], built[node]) for node in built)

    def test_a_written_sparse_draw_reads_back_as_the_registry_built_it(
        self, tmp_path: Path
    ) -> None:
        draws = _draws(SPARSE_KEY, _sparse, WIDTH)

        write_seed(SeedContext(draws=draws, output_root=tmp_path), 3)

        path = dataset_target(SPARSE_KEY, tmp_path, 3)
        written = _written(path)
        built = draw(draws, 3)
        assert sorted(written) == sorted(built)
        assert all(
            np.array_equal(written[node][0], built[node][0])
            and np.array_equal(written[node][1], built[node][1])
            for node in built
        )
        assert pq.read_table(path).schema.metadata[b"dim"] == str(WIDTH).encode()

    def test_records_the_family_and_seed_the_draw_came_from(self, tmp_path: Path) -> None:
        write_seed(SeedContext(draws=_draws(DENSE_KEY, _dense, None), output_root=tmp_path), 3)

        metadata = pq.read_table(dataset_target(DENSE_KEY, tmp_path, 3)).schema.metadata

        assert metadata[b"description"].decode() == (
            f"Shuffle-null order-effect control for {DENSE_KEY}, seed 3."
        )


class TestGenerate:
    def test_writes_one_dataset_per_seed(self, tmp_path: Path) -> None:
        written = generate(_draws(DENSE_KEY, _dense, None), tmp_path, 3, max_workers=1)

        assert written == [
            "construction=icf_position4_shuffle0001",
            "construction=icf_position4_shuffle0002",
            "construction=icf_position4_shuffle0003",
        ]
        assert all(dataset_target(DENSE_KEY, tmp_path, seed).exists() for seed in (1, 2, 3))

    def test_a_seed_writes_the_draw_its_own_permutation_produces(self, tmp_path: Path) -> None:
        """A generator that ignored its seed would write one null a thousand times."""
        generate(_draws(DENSE_KEY, _dense, None), tmp_path, 2, max_workers=1)

        first = _written(dataset_target(DENSE_KEY, tmp_path, 1))
        second = _written(dataset_target(DENSE_KEY, tmp_path, 2))

        assert not all(np.array_equal(first[node], second[node]) for node in first)


class TestMain:
    def test_writes_every_seed_of_the_family_it_is_given(self, tmp_path: Path) -> None:
        main(
            [
                "--family",
                DENSE_KEY,
                "--output-root",
                str(tmp_path),
                "--config-root",
                str(tmp_path),
                "--n-shuffles",
                "2",
                "--max-workers",
                "1",
            ],
            draws_factory=lambda key, _config_root: _draws(key, _dense, None),
        )

        assert [dataset_target(DENSE_KEY, tmp_path, seed).exists() for seed in (1, 2)] == [
            True,
            True,
        ]


class TestSeedNaming:
    """A draw is found by its seeded name, so the width and its refusal are the contract."""

    @pytest.mark.parametrize(
        ("seed", "expected"),
        [
            (9, "construction=icf_position4_shuffle0009"),
            (99, "construction=icf_position4_shuffle0099"),
            (100, "construction=icf_position4_shuffle0100"),
            (999, "construction=icf_position4_shuffle0999"),
            (DEFAULT_N_SHUFFLES, "construction=icf_position4_shuffle1000"),
        ],
    )
    def test_a_high_seed_writes_under_its_padded_name(
        self, seed: int, expected: str, tmp_path: Path
    ) -> None:
        context = SeedContext(draws=_draws(DENSE_KEY, _dense, None), output_root=tmp_path)

        assert write_seed(context, seed) == expected
        assert dataset_target(DENSE_KEY, tmp_path, seed).exists()

    def test_names_written_across_the_hundred_boundary_sort_in_seed_order(
        self, tmp_path: Path
    ) -> None:
        context = SeedContext(draws=_draws(DENSE_KEY, _dense, None), output_root=tmp_path)

        names = [write_seed(context, seed) for seed in (1, 9, 10, 99, 100, 101, 999, 1000)]

        assert names == sorted(names)

    def test_a_seed_past_the_name_width_is_refused_before_anything_is_written(
        self, tmp_path: Path
    ) -> None:
        context = SeedContext(draws=_draws(DENSE_KEY, _dense, None), output_root=tmp_path)

        with pytest.raises(ValueError, match="exceeds"):
            write_seed(context, 10000)

        assert list(tmp_path.rglob("*.parquet")) == []


class TestShuffleCount:
    def test_the_cli_defaults_to_the_shared_shuffle_count(self, tmp_path: Path) -> None:
        args = build_parser().parse_args(
            ["--family", DENSE_KEY, "--output-root", str(tmp_path), "--config-root", str(tmp_path)]
        )

        assert args.n_shuffles == DEFAULT_N_SHUFFLES

    def test_the_cli_lets_the_shuffle_count_be_overridden(self, tmp_path: Path) -> None:
        args = build_parser().parse_args(
            [
                "--family",
                DENSE_KEY,
                "--output-root",
                str(tmp_path),
                "--config-root",
                str(tmp_path),
                "--n-shuffles",
                "7",
            ]
        )

        assert args.n_shuffles == 7


class TestControlPairing:
    """A shuffle-null must land beside the dataset it controls for, or it controls for nothing."""

    @pytest.mark.parametrize("key", sorted(FAMILIES))
    def test_a_draw_differs_from_its_real_dataset_only_by_the_construction_the_seed_names(
        self, key: str
    ) -> None:
        root = Path("/out")
        construction = key.rsplit("/", 1)[1]

        seeded = dataset_target(key, root, 1)

        assert seeded.parent.name == f"construction={construction}_shuffle0001"
        assert seeded.parent.parent == dataset_target(key, root, 2).parent.parent

    @pytest.mark.parametrize("key", sorted(FAMILIES))
    def test_the_unshuffled_dataset_sits_under_the_partition_its_draws_do(self, key: str) -> None:
        """A control scores a family against the real dataset, so both must be one resolution."""
        root = Path("/out")
        construction = key.rsplit("/", 1)[1]

        real = dataset_source(key, root)

        assert real.parent.name == f"construction={construction}"
        assert real.parent.parent == dataset_target(key, root, 1).parent.parent

    @pytest.mark.parametrize("key", sorted(FAMILIES))
    @pytest.mark.integration
    def test_the_unshuffled_dataset_exists_in_the_committed_tree(self, key: str) -> None:
        """A path convention that no longer matches the tree would silently score nothing."""
        assert dataset_source(key, EMBEDDINGS_DATA).exists()

    @pytest.mark.parametrize(
        ("key", "unit", "domain", "level", "construction"),
        [
            (
                "morphological/morph_suffix/posmean",
                SUFFIX_UNIT,
                MORPHOLOGICAL_DOMAIN,
                None,
                MORPHOLOGICAL_CONSTRUCTION,
            ),
            (
                "syntactic/phrase/signature/posmean",
                SIGNATURE_UNIT,
                SYNTACTIC_DOMAIN,
                "phrase",
                SYNTACTIC_CONSTRUCTION,
            ),
        ],
    )
    def test_a_deploy_control_lands_beside_the_dataset_its_generator_writes(
        self, key: str, unit: str, domain: str, level: str | None, construction: str
    ) -> None:
        """The registry key names the unit and construction, so it must name the real one's."""
        root = Path("/out")
        real = dataset_path(
            root, unit, construction, domain=domain, unit_key="feature", level=level
        )

        assert key.endswith(f"/{construction}")
        assert dataset_target(key, root, 1).parent.parent == real.parent.parent
