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
        (_, vocalized, niqqud_only, _) = variations_for_model("miqrabert")[0]
        assert vocalized is False
        assert niqqud_only is False


class TestDatasetName:
    def test_combines_the_name_slug_and_variation(self):
        assert dataset_name("bge-m3", "vocalized") == "semantic_bge_m3_vocalized"

    def test_uses_the_full_technical_slug_not_the_registry_key(self):
        assert (
            dataset_name("gemini", "cantillation") == "semantic_gemini_embedding_2_cantillation"
        )


class TestDatasetDescription:
    def test_combines_model_and_variation_descriptions_and_the_schema_note(self):
        description = dataset_description("bge-m3", "Bare consonants.")
        assert MODEL_REGISTRY["bge-m3"][2] in description
        assert "Bare consonants." in description
        assert "node_id" in description


class TestModelRegistryAndVariationsAreConsistent:
    def test_every_model_registry_entry_has_three_fields(self):
        for slug, entry in MODEL_REGISTRY.items():
            assert len(entry) == 3, slug

    def test_every_variation_entry_has_four_fields(self):
        for variation in VARIATIONS:
            assert len(variation) == 4

    def test_every_diacritic_stripping_slug_is_a_real_model_registry_key(self):
        for slug in TOKENIZER_STRIPS_ALL_DIACRITICS:
            assert slug in MODEL_REGISTRY
