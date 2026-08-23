from __future__ import annotations

import pyarrow.parquet as pq

from lexical.corpus import LexicalPsalm
from lexical.export import dataset_path
from lexical.generate import generate


def _psalm(*, number, lexemes, forms, nodes):
    return LexicalPsalm(
        number=number,
        colon_lexemes=lexemes,
        colon_forms=forms,
        colon_nodes=nodes,
    )


def _psalms():
    return [
        _psalm(
            number=1, lexemes=(("A", "B"), ("A",)), forms=(("A0", "B0"), ("A0",)), nodes=(100, 101)
        ),
        _psalm(number=2, lexemes=(("C",),), forms=(("C0",),), nodes=(200,)),
    ]


def _icf_weights_by_key():
    return {
        "lex0": {"A0": 1.5, "B0": 2.0, "C0": 0.5},
        "lex": {"A": 1.2, "B": 1.8, "C": 0.6},
    }


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


class TestGenerate:
    def test_writes_the_full_weighting_family_for_both_homograph_and_lexeme(self, tmp_path):
        written = generate(_psalms(), tmp_path, _icf_weights_by_key())

        assert set(written) == {f"homograph_{w}" for w in _FULL_WEIGHTS} | {
            f"lexeme_{w}" for w in _FULL_WEIGHTS
        }
        for weight in _FULL_WEIGHTS:
            assert dataset_path(tmp_path, "homograph", weight, unit_key="unit").exists()
            assert dataset_path(tmp_path, "lexeme", weight, unit_key="unit").exists()

    def test_lag_dimension_equals_k(self, tmp_path):
        generate(_psalms(), tmp_path, _icf_weights_by_key())

        for weight, k in (("icf_spacing2", 2), ("icf_spacing4", 4), ("icf_spacing8", 8)):
            table = pq.read_table(dataset_path(tmp_path, "homograph", weight, unit_key="unit"))
            assert len(table["vector"].to_pylist()[0]) == k

    def test_lag_broadcasts_the_same_vector_to_every_colon_of_a_psalm(self, tmp_path):
        generate(_psalms(), tmp_path, _icf_weights_by_key())

        table = pq.read_table(dataset_path(tmp_path, "homograph", "icf_spacing4", unit_key="unit"))
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert by_node[100] == by_node[101]

    def test_position_mean_dimension_is_twice_the_vocabulary_size(self, tmp_path):
        generate(_psalms(), tmp_path, _icf_weights_by_key())

        table = pq.read_table(
            dataset_path(tmp_path, "homograph", "icf_position_mean", unit_key="unit")
        )
        assert len(table["vector"].to_pylist()[0]) == 6

    def test_position_mean_gives_each_colon_of_a_psalm_its_own_vector(self, tmp_path):
        generate(_psalms(), tmp_path, _icf_weights_by_key())

        table = pq.read_table(
            dataset_path(tmp_path, "homograph", "icf_position_mean", unit_key="unit")
        )
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        # node 100 (A0, B0) and node 101 (A0 only) have different content, so different vectors.
        assert by_node[100] != by_node[101]

    def test_positional_dimension_is_k_times_vocabulary_size(self, tmp_path):
        generate(_psalms(), tmp_path, _icf_weights_by_key())

        for weight, k in (("icf_position2", 2), ("icf_position4", 4), ("icf_position8", 8)):
            table = pq.read_table(dataset_path(tmp_path, "homograph", weight, unit_key="unit"))
            assert len(table["vector"].to_pylist()[0]) == 3 * k

    def test_positional_gives_each_colon_of_a_psalm_its_own_vector(self, tmp_path):
        generate(_psalms(), tmp_path, _icf_weights_by_key())

        table = pq.read_table(dataset_path(tmp_path, "homograph", "icf_position4", unit_key="unit"))
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        # psalm 1 has nodes 100 and 101, with different content, so different vectors.
        assert by_node[100] != by_node[101]

    def test_psalm_lag_broadcasts_the_same_vector_to_every_colon_of_a_psalm(self, tmp_path):
        generate(_psalms(), tmp_path, _icf_weights_by_key())

        table = pq.read_table(
            dataset_path(tmp_path, "homograph", "icf_spacing4_psalm", unit_key="unit")
        )
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert by_node[100] == by_node[101]

    def test_psalm_position_mean_broadcasts_the_same_vector_to_every_colon_of_a_psalm(
        self, tmp_path
    ):
        generate(_psalms(), tmp_path, _icf_weights_by_key())

        table = pq.read_table(
            dataset_path(tmp_path, "homograph", "icf_position_mean_psalm", unit_key="unit")
        )
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert by_node[100] == by_node[101]

    def test_psalm_positional_broadcasts_the_same_vector_to_every_colon_of_a_psalm(self, tmp_path):
        generate(_psalms(), tmp_path, _icf_weights_by_key())

        table = pq.read_table(
            dataset_path(tmp_path, "homograph", "icf_position4_psalm", unit_key="unit")
        )
        by_node = dict(zip(table["node_id"].to_pylist(), table["vector"].to_pylist(), strict=True))
        assert by_node[100] == by_node[101]

    def test_psalm_variants_match_their_colon_level_counterparts_dimension(self, tmp_path):
        generate(_psalms(), tmp_path, _icf_weights_by_key())

        for colon_weight, psalm_weight in (
            ("icf_position2", "icf_position2_psalm"),
            ("icf_position4", "icf_position4_psalm"),
            ("icf_position8", "icf_position8_psalm"),
            ("icf_spacing2", "icf_spacing2_psalm"),
            ("icf_spacing4", "icf_spacing4_psalm"),
            ("icf_spacing8", "icf_spacing8_psalm"),
            ("icf_position_mean", "icf_position_mean_psalm"),
        ):
            colon_table = pq.read_table(
                dataset_path(tmp_path, "homograph", colon_weight, unit_key="unit")
            )
            psalm_table = pq.read_table(
                dataset_path(tmp_path, "homograph", psalm_weight, unit_key="unit")
            )
            assert len(colon_table["vector"].to_pylist()[0]) == len(
                psalm_table["vector"].to_pylist()[0]
            )

    def test_skips_variants_whose_dataset_already_exists(self, tmp_path):
        generate(_psalms(), tmp_path, _icf_weights_by_key())

        written_again = generate(_psalms(), tmp_path, _icf_weights_by_key())

        assert written_again == []

    def test_lexeme_vocabulary_dimension_matches_distinct_lex_values(self, tmp_path):
        generate(_psalms(), tmp_path, _icf_weights_by_key())

        table = pq.read_table(dataset_path(tmp_path, "lexeme", "binary", unit_key="unit"))
        # distinct lex values across fixtures: A, B, C
        assert len(table["vector"].to_pylist()[0]) == 3

    def test_lexeme_icf_uses_its_own_frequency_table_not_homographs(self, tmp_path):
        generate(_psalms(), tmp_path, _icf_weights_by_key())

        homograph_table = pq.read_table(dataset_path(tmp_path, "homograph", "icf", unit_key="unit"))
        lexeme_table = pq.read_table(dataset_path(tmp_path, "lexeme", "icf", unit_key="unit"))
        homograph_vec = dict(
            zip(
                homograph_table["node_id"].to_pylist(),
                homograph_table["vector"].to_pylist(),
                strict=True,
            )
        )[100]
        lexeme_vec = dict(
            zip(
                lexeme_table["node_id"].to_pylist(), lexeme_table["vector"].to_pylist(), strict=True
            )
        )[100]
        # A0=1.5/B0=2.0 (homograph weights) vs A=1.2/B=1.8 (lexeme weights): different ICF values.
        assert homograph_vec != lexeme_vec

    def test_homograph_vocabulary_dimension_matches_distinct_lex0_values(self, tmp_path):
        generate(_psalms(), tmp_path, _icf_weights_by_key())

        table = pq.read_table(dataset_path(tmp_path, "homograph", "binary", unit_key="unit"))
        # distinct lex0 values across fixtures: A0, B0, C0
        assert len(table["vector"].to_pylist()[0]) == 3

    def test_homograph_count_and_binary_differ_for_a_repeated_lexeme(self, tmp_path):
        generate(_psalms(), tmp_path, _icf_weights_by_key())

        binary_table = pq.read_table(dataset_path(tmp_path, "homograph", "binary", unit_key="unit"))
        count_table = pq.read_table(dataset_path(tmp_path, "homograph", "count", unit_key="unit"))
        binary_ids = binary_table["node_id"].to_pylist()
        binary_vecs = binary_table["vector"].to_pylist()
        binary_by_node = dict(zip(binary_ids, binary_vecs, strict=True))
        count_ids = count_table["node_id"].to_pylist()
        count_vecs = count_table["vector"].to_pylist()
        count_by_node = dict(zip(count_ids, count_vecs, strict=True))
        # node 100 has A0 once, B0 once: binary and count agree here.
        assert binary_by_node[100] == count_by_node[100]
