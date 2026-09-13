"""Unit tests for generate.py: every external dependency is injected, no network or model loads."""

from __future__ import annotations

import numpy as np
import pytest

from semantic.export import dataset_path
from semantic.generate import generate_api, generate_local


def _psalm(*, number: int, half_verses, half_verses_unvocalized, half_verses_niqqud_only=()):
    from semantic.corpus import SemanticPsalm

    return SemanticPsalm(
        number=number,
        half_verses=half_verses,
        half_verses_unvocalized=half_verses_unvocalized,
        half_verses_niqqud_only=half_verses_niqqud_only,
        half_verse_nodes=tuple(range(number * 100, number * 100 + len(half_verses))),
    )


class TestGenerateLocal:
    def test_computes_and_writes_every_variation_for_a_diacritic_preserving_model(self, tmp_path):
        calls = []

        def _fake_compute(psalms, model_name, *, tier, device, torch_dtype):
            calls.append((model_name, tier))
            return {p.number: np.zeros((len(p.half_verses), 2)) for p in psalms}

        psalms = [_psalm(number=1, half_verses=("A",), half_verses_unvocalized=("a",))]

        written = generate_local(psalms, tmp_path, "bge-m3", compute=_fake_compute)

        assert written == [
            "semantic_bge_m3_consonantal",
            "semantic_bge_m3_vocalized",
            "semantic_bge_m3_cantillation",
        ]
        assert [tier for _, tier in calls] == ["consonantal", "vocalized", "cantillation"]
        for variation in ("consonantal", "vocalized", "cantillation"):
            assert dataset_path(tmp_path, "bge_m3", variation).exists()

    def test_computes_only_one_variation_for_a_diacritic_stripping_model(self, tmp_path):
        calls = []

        def _fake_compute(psalms, model_name, *, tier, device, torch_dtype):
            calls.append((model_name, tier))
            return {p.number: np.zeros((len(p.half_verses), 2)) for p in psalms}

        psalms = [_psalm(number=1, half_verses=("A",), half_verses_unvocalized=("a",))]

        written = generate_local(psalms, tmp_path, "miqrabert", compute=_fake_compute)

        assert written == ["semantic_miqrabert_consonantal"]
        assert calls == [("davidmsmiley/MiqraBERT", "consonantal")]

    def test_restricts_to_a_single_named_variation(self, tmp_path):
        calls = []

        def _fake_compute(psalms, model_name, *, tier, device, torch_dtype):
            calls.append(tier)
            return {p.number: np.zeros((len(p.half_verses), 2)) for p in psalms}

        psalms = [_psalm(number=1, half_verses=("A",), half_verses_unvocalized=("a",))]

        written = generate_local(
            psalms, tmp_path, "bge-m3", variation="vocalized", compute=_fake_compute
        )

        assert written == ["semantic_bge_m3_vocalized"]
        assert len(calls) == 1

    def test_a_variation_not_offered_for_the_model_writes_nothing(self, tmp_path):
        def _must_not_be_called(*args, **kwargs):
            raise AssertionError("compute must not be called for an unavailable variation")

        psalms = [_psalm(number=1, half_verses=("A",), half_verses_unvocalized=("a",))]

        written = generate_local(
            psalms, tmp_path, "miqrabert", variation="vocalized", compute=_must_not_be_called
        )

        assert written == []

    def test_skips_a_variation_whose_tf_file_already_exists(self, tmp_path):
        from semantic.export import node_vectors, write_dataset

        psalms = [_psalm(number=1, half_verses=("A",), half_verses_unvocalized=("a",))]
        write_dataset(
            tmp_path,
            "miqrabert",
            "consonantal",
            node_vectors({1: np.zeros((1, 2))}, psalms),
            "already here",
        )

        def _must_not_be_called(*args, **kwargs):
            raise AssertionError("compute must not be called for an already-written variation")

        written = generate_local(psalms, tmp_path, "miqrabert", compute=_must_not_be_called)

        assert written == []

    def test_passes_device_and_torch_dtype_through(self, tmp_path):
        calls = []

        def _fake_compute(psalms, model_name, *, tier, device, torch_dtype):
            calls.append((device, torch_dtype))
            return {p.number: np.zeros((len(p.half_verses), 2)) for p in psalms}

        psalms = [_psalm(number=1, half_verses=("A",), half_verses_unvocalized=("a",))]

        generate_local(
            psalms,
            tmp_path,
            "miqrabert",
            device="cuda",
            torch_dtype="bfloat16",
            compute=_fake_compute,
        )

        assert calls == [("cuda", "bfloat16")]


class TestGenerateApi:
    def test_cache_miss_calls_fetch_once_per_variation_with_flattened_half_verses(self, tmp_path):
        calls = []

        def _fake_fetch(texts, *, api_key):
            calls.append((list(texts), api_key))
            return np.zeros((len(texts), 3))

        psalms = [
            _psalm(
                number=1,
                half_verses=("a1", "a2"),
                half_verses_unvocalized=("u1", "u2"),
                half_verses_niqqud_only=("n1", "n2"),
            )
        ]

        written = generate_api(
            psalms,
            tmp_path,
            "gemini",
            fetch=_fake_fetch,
            env={"TEHILLIM_OPENROUTER_API_KEY": "test-key"},
        )

        assert written == [
            "semantic_gemini_embedding_2_consonantal",
            "semantic_gemini_embedding_2_vocalized",
            "semantic_gemini_embedding_2_cantillation",
        ]
        assert len(calls) == 3
        texts_by_call = [c[0] for c in calls]
        assert ["u1", "u2"] in texts_by_call
        assert ["n1", "n2"] in texts_by_call
        assert ["a1", "a2"] in texts_by_call
        assert all(api_key == "test-key" for _, api_key in calls)

    def test_cache_hit_never_calls_fetch(self, tmp_path):
        from semantic.export import node_vectors, write_dataset

        psalms = [_psalm(number=1, half_verses=("A",), half_verses_unvocalized=("a",))]
        for variation in ("consonantal", "vocalized", "cantillation"):
            write_dataset(
                tmp_path,
                "gemini_embedding_2",
                variation,
                node_vectors({1: np.zeros((1, 3))}, psalms),
                "already here",
            )

        def _must_not_be_called(texts, *, api_key):
            raise AssertionError("fetch must not be called on a cache hit")

        written = generate_api(psalms, tmp_path, "gemini", fetch=_must_not_be_called, env={})

        assert written == []

    def test_missing_api_key_raises_naming_the_exact_env_var(self, tmp_path):
        psalms = [_psalm(number=1, half_verses=("A",), half_verses_unvocalized=("a",))]

        def _fake_fetch(texts, *, api_key):
            return np.zeros((1, 1))

        with pytest.raises(RuntimeError, match="TEHILLIM_OPENROUTER_API_KEY"):
            generate_api(psalms, tmp_path, "gemini", fetch=_fake_fetch, env={})

    def test_missing_api_key_is_not_read_when_every_variation_is_already_cached(self, tmp_path):
        from semantic.export import node_vectors, write_dataset

        psalms = [_psalm(number=1, half_verses=("A",), half_verses_unvocalized=("a",))]
        for variation in ("consonantal", "vocalized", "cantillation"):
            write_dataset(
                tmp_path,
                "gemini_embedding_2",
                variation,
                node_vectors({1: np.zeros((1, 3))}, psalms),
                "already here",
            )

        def _must_not_be_called(texts, *, api_key):
            raise AssertionError("fetch must not be called when every variation is cached")

        written = generate_api(psalms, tmp_path, "gemini", fetch=_must_not_be_called, env={})

        assert written == []

    def test_cohere_uses_a_separate_api_key_env_var(self, tmp_path):
        calls = []

        def _fake_fetch(texts, *, api_key):
            calls.append(api_key)
            return np.zeros((len(texts), 2))

        psalms = [
            _psalm(
                number=1,
                half_verses=("A",),
                half_verses_unvocalized=("a",),
                half_verses_niqqud_only=("n",),
            )
        ]

        generate_api(
            psalms,
            tmp_path,
            "cohere",
            fetch=_fake_fetch,
            env={"TEHILLIM_COHERE_API_KEY": "cohere-key"},
        )

        assert calls
        assert all(key == "cohere-key" for key in calls)


class TestSemanticSpecs:
    def test_one_spec_per_registered_model_covering_its_text_variations(self) -> None:
        """Each model declares exactly the text partitions its tokenizer distinguishes."""
        from semantic.generate import SPECS
        from semantic.registry import MODEL_REGISTRY, variations_for_model

        by_slug = {spec.module.rsplit(":", 1)[1]: spec for spec in SPECS}
        assert set(by_slug) == set(MODEL_REGISTRY)
        for slug, spec in by_slug.items():
            model_slug = MODEL_REGISTRY[slug][1]
            expected = tuple(
                f"domain=semantic/model={model_slug}/text={tier}"
                for tier, _ in variations_for_model(slug)
            )
            assert spec.partitions == expected

    def test_resource_classes_separate_hosted_large_and_small_models(self) -> None:
        """Hosted models need an API, Colab-only models a GPU, the rest run on the CPU."""
        from semantic.generate import SPECS, resource_for

        assert resource_for("gemini") == "api"
        assert resource_for("kalm-embedding") == "gpu"
        assert resource_for("berel") is None
        assert {spec.resource for spec in SPECS} == {"api", "gpu", None}


class TestSemanticMainModelFilter:
    def test_model_flag_generates_only_that_model(self, tmp_path) -> None:
        """`--model` restricts a run to one registry slug so a driver cell is one model."""
        from semantic import generate as module

        calls: list[str] = []

        def fake_local(psalms, output_root, slug, **_):
            calls.append(slug)
            return [slug]

        def fake_api(psalms, output_root, slug, **_):
            calls.append(slug)
            return [slug]

        class FakeCorpus:
            def psalms(self):
                return []

        module.main(
            ["--output-root", str(tmp_path), "--model", "berel"],
            corpus_factory=FakeCorpus,
            local=fake_local,
            api=fake_api,
        )
        assert calls == ["berel"]

    def test_without_model_flag_every_model_runs(self, tmp_path) -> None:
        """No flag keeps the historical behaviour of writing every registered model."""
        from semantic import generate as module
        from semantic.registry import MODEL_REGISTRY

        calls: list[str] = []

        def fake(psalms, output_root, slug, **_):
            calls.append(slug)
            return [slug]

        class FakeCorpus:
            def psalms(self):
                return []

        module.main(
            ["--output-root", str(tmp_path)], corpus_factory=FakeCorpus, local=fake, api=fake
        )
        assert calls == list(MODEL_REGISTRY)


class TestLargeModelDtype:
    def test_large_models_load_in_their_registered_dtype(self, tmp_path) -> None:
        """A GPU cell passes the notebook's dtype so the vectors match the Colab runs."""
        from semantic import generate as module

        seen: dict[str, object] = {}

        def fake_local(psalms, output_root, slug, **kwargs):
            seen[slug] = kwargs.get("torch_dtype")
            return [slug]

        module.generate([], tmp_path, ("kalm-embedding", "berel"), local=fake_local, api=fake_local)
        assert seen == {"kalm-embedding": "bfloat16", "berel": None}
