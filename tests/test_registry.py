from __future__ import annotations

from semantic.registry import (
    MODEL_REGISTRY,
    TOKENIZER_STRIPS_ALL_DIACRITICS,
    VARIATIONS,
    dataset_description,
    dataset_name,
    variations_for_model,
)


class TestVariationsForModel:
    def test_returns_all_three_variations_for_a_diacritic_preserving_model(self):
        variations = variations_for_model("bge-m3")
        assert [t[0] for t in variations] == ["consonantal", "vocalized", "cantillation"]

    def test_returns_only_consonantal_for_a_diacritic_stripping_model(self):
        variations = variations_for_model("miqrabert")
        assert [t[0] for t in variations] == ["consonantal"]

    def test_diacritic_stripping_model_is_fed_consonantal_text_directly(self):
        (tier, _) = variations_for_model("miqrabert")[0]
        assert tier == "consonantal"


class TestDatasetName:
    def test_combines_the_name_slug_and_variation(self):
        assert dataset_name("bge-m3", "vocalized") == "semantic_bge_m3_vocalized"

    def test_uses_the_full_technical_slug_not_the_registry_key(self):
        assert dataset_name("gemini", "cantillation") == "semantic_gemini_embedding_2_cantillation"


class TestDatasetDescription:
    def test_combines_model_and_variation_descriptions_and_the_schema_note(self):
        description = dataset_description("bge-m3", "Bare consonants.")
        assert MODEL_REGISTRY["bge-m3"][2] in description
        assert "Bare consonants." in description
        assert "node_id" in description


class TestNewLargeModelsAreRegistered:
    def test_harrier_oss_v1_uses_the_verified_hugging_face_id(self):
        assert MODEL_REGISTRY["harrier-oss-v1"][0] == "microsoft/harrier-oss-v1-27b"

    def test_f2llm_v2_uses_the_verified_hugging_face_id(self):
        assert MODEL_REGISTRY["f2llm-v2"][0] == "codefuse-ai/F2LLM-v2-14B"

    def test_both_get_all_three_text_variations(self):
        assert [t[0] for t in variations_for_model("harrier-oss-v1")] == [
            "consonantal",
            "vocalized",
            "cantillation",
        ]
        assert [t[0] for t in variations_for_model("f2llm-v2")] == [
            "consonantal",
            "vocalized",
            "cantillation",
        ]


class TestModelRegistryAndVariationsAreConsistent:
    def test_every_model_registry_entry_has_three_fields(self):
        for slug, entry in MODEL_REGISTRY.items():
            assert len(entry) == 3, slug

    def test_every_variation_entry_is_a_tier_and_its_description(self):
        for variation in VARIATIONS:
            assert len(variation) == 2

    def test_every_variation_names_one_of_the_three_text_tiers(self):
        tiers = [tier for tier, _ in VARIATIONS]

        assert tiers == ["consonantal", "vocalized", "cantillation"]

    def test_every_diacritic_stripping_slug_is_a_real_model_registry_key(self):
        for slug in TOKENIZER_STRIPS_ALL_DIACRITICS:
            assert slug in MODEL_REGISTRY


class TestRegistryCoversEveryKnownModel:
    """A model constant that never reaches MODEL_REGISTRY is a model nothing can generate."""

    def test_every_local_model_constant_is_registered(self):
        from semantic import local_models

        constants = {
            value
            for name, value in vars(local_models).items()
            if name.endswith("_MODEL") and isinstance(value, str)
        }
        registered = {technical for technical, _, _ in MODEL_REGISTRY.values()}

        assert constants - registered == set()

    def test_every_api_model_constant_is_registered(self):
        from semantic import api_models

        constants = {
            value
            for name, value in vars(api_models).items()
            if name.endswith("_MODEL") and isinstance(value, str)
        }
        registered = {technical for technical, _, _ in MODEL_REGISTRY.values()}

        assert constants - registered == set()

    def test_technical_ids_are_unique_across_the_registry(self):
        technical = [entry[0] for entry in MODEL_REGISTRY.values()]

        assert len(technical) == len(set(technical))

    def test_dataset_slugs_are_unique_across_the_registry(self):
        slugs = [entry[1] for entry in MODEL_REGISTRY.values()]

        assert len(slugs) == len(set(slugs))
