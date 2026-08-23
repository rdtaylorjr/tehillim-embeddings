from __future__ import annotations

import pyarrow.parquet as pq

from lexical.export import dataset_path
from lexical.generate_surface import generate_surface
from lexical.surface_corpus import SurfacePsalm

_TIERS = ("consonantal", "vocalized", "cantillation")

_FULL_WEIGHTS = (
    "binary",
    "count",
    "log_count",
    "icf",
    "tf_icf",
    "icf_position2",
    "icf_position4",
    "icf_position8",
    "icf_spacing2",
    "icf_spacing4",
    "icf_spacing8",
    "icf_position_mean",
    "icf_position2_psalm",
    "icf_position4_psalm",
    "icf_position8_psalm",
    "icf_spacing2_psalm",
    "icf_spacing4_psalm",
    "icf_spacing8_psalm",
    "icf_position_mean_psalm",
)


def _psalm(*, number, consonantal, nodes):
    return SurfacePsalm(
        number=number,
        colon_consonantal=consonantal,
        colon_vocalized=consonantal,
        colon_cantillation=consonantal,
        colon_nodes=nodes,
    )


def _psalms():
    return [
        _psalm(number=1, consonantal=(("א", "ב"), ("א",)), nodes=(100, 101)),
        _psalm(number=2, consonantal=(("ג",),), nodes=(200,)),
    ]


def _icf_weights_by_tier():
    weights = {"א": 1.5, "ב": 2.0, "ג": 0.5}
    return {tier: weights for tier in _TIERS}


class TestGenerateSurface:
    def test_writes_the_full_weighting_family_for_all_three_text_tiers(self, tmp_path):
        written = generate_surface(_psalms(), tmp_path, _icf_weights_by_tier())

        assert set(written) == {
            f"word_{tier}_{weight}" for tier in _TIERS for weight in _FULL_WEIGHTS
        }
        for tier in _TIERS:
            for weight in _FULL_WEIGHTS:
                assert dataset_path(tmp_path, "word", weight, text=tier, unit_key="unit").exists()

    def test_vocabulary_dimension_matches_distinct_forms(self, tmp_path):
        generate_surface(_psalms(), tmp_path, _icf_weights_by_tier())

        table = pq.read_table(
            dataset_path(tmp_path, "word", "binary", text="consonantal", unit_key="unit")
        )
        # distinct forms across fixtures: א, ב, ג
        assert len(table["vector"].to_pylist()[0]) == 3

    def test_positional_dimension_is_k_times_vocabulary_size(self, tmp_path):
        generate_surface(_psalms(), tmp_path, _icf_weights_by_tier())

        table = pq.read_table(
            dataset_path(tmp_path, "word", "icf_position4", text="consonantal", unit_key="unit")
        )
        assert len(table["vector"].to_pylist()[0]) == 3 * 4

    def test_position_mean_dimension_is_twice_the_vocabulary_size(self, tmp_path):
        generate_surface(_psalms(), tmp_path, _icf_weights_by_tier())

        table = pq.read_table(
            dataset_path(tmp_path, "word", "icf_position_mean", text="consonantal", unit_key="unit")
        )
        assert len(table["vector"].to_pylist()[0]) == 6

    def test_psalm_variant_broadcasts_the_same_vector_to_every_colon(self, tmp_path):
        generate_surface(_psalms(), tmp_path, _icf_weights_by_tier())

        table = pq.read_table(
            dataset_path(
                tmp_path, "word", "icf_position4_psalm", text="consonantal", unit_key="unit"
            )
        )
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert by_node[100] == by_node[101]

    def test_skips_variants_whose_dataset_already_exists(self, tmp_path):
        generate_surface(_psalms(), tmp_path, _icf_weights_by_tier())

        written_again = generate_surface(_psalms(), tmp_path, _icf_weights_by_tier())

        assert written_again == []
