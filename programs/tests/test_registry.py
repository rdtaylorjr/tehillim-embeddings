from __future__ import annotations

import base64

import numpy as np

from tehillim_embeddings.registry import (
    MODEL_REGISTRY,
    TIERS,
    encode_vector,
    feature_description,
    feature_name,
    tiers_for_model,
)


class TestTiersForModel:
    def test_returns_all_three_tiers_for_a_diacritic_preserving_model(self):
        tiers = tiers_for_model("bge-m3")
        assert [t[0] for t in tiers] == ["consonantal", "vocalized", "cantillation"]

    def test_returns_only_consonantal_for_a_diacritic_stripping_model(self):
        tiers = tiers_for_model("miqrabert")
        assert [t[0] for t in tiers] == ["consonantal"]

    def test_diacritic_stripping_model_still_encodes_vocalized_input(self):
        # These models' input is fed as fully vocalized text even though
        # the output is labeled consonantal - the tokenizer, not the
        # input text, is what strips the signal.
        (_, vocalized, niqqud_only, _) = tiers_for_model("miqrabert")[0]
        assert vocalized is True
        assert niqqud_only is False


class TestFeatureName:
    def test_combines_the_feature_slug_and_tier(self):
        assert feature_name("bge-m3", "vocalized") == "semantic_bge_m3_vocalized"

    def test_uses_the_full_technical_slug_not_the_registry_key(self):
        assert (
            feature_name("gemini", "cantillation") == "semantic_gemini_embedding_2_cantillation"
        )


class TestFeatureDescription:
    def test_combines_model_and_tier_descriptions_and_the_encoding_note(self):
        description = feature_description("bge-m3", "Bare consonants.")
        assert MODEL_REGISTRY["bge-m3"][2] in description
        assert "Bare consonants." in description
        assert "base64" in description


class TestEncodeVector:
    def test_round_trips_through_base64(self):
        vector = np.array([1.0, -2.5, 0.0], dtype=np.float32)
        encoded = encode_vector(vector)
        decoded = np.frombuffer(base64.b64decode(encoded), dtype="<f4")
        assert np.array_equal(decoded, vector)

    def test_casts_to_float32_before_encoding(self):
        vector = np.array([1.0, 2.0], dtype=np.float64)
        encoded = encode_vector(vector)
        decoded = np.frombuffer(base64.b64decode(encoded), dtype="<f4")
        assert decoded.dtype == np.float32


class TestModelRegistryAndTiersAreConsistent:
    def test_every_model_registry_entry_has_three_fields(self):
        for slug, entry in MODEL_REGISTRY.items():
            assert len(entry) == 3, slug

    def test_every_tier_entry_has_four_fields(self):
        for tier in TIERS:
            assert len(tier) == 4
