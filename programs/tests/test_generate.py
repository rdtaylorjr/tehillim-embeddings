"""Unit tests for generate.py's orchestration: which (model, tier)
pairs get computed, the .tf-file-exists cache check, and API key
handling. No monkeypatch: `compute`/`fetch` are passed as explicit
arguments. Corpus/Psalm loading is exercised with hand-built Psalm
objects, no real BHSA load.
"""

from __future__ import annotations

import numpy as np
import pytest

from tehillim_embeddings.export import feature_path
from tehillim_embeddings.generate import generate_api, generate_local


def _psalm(*, number: int, half_verses, half_verses_unvocalized, half_verses_niqqud_only=()):
    from tehillim_pipeline.corpus import Psalm

    return Psalm(
        number=number,
        verse_count=1,
        words=(),
        incipit="",
        half_verses=half_verses,
        half_verses_unvocalized=half_verses_unvocalized,
        half_verses_niqqud_only=half_verses_niqqud_only,
        half_verse_nodes=tuple(range(number * 100, number * 100 + len(half_verses))),
    )


class TestGenerateLocal:
    def test_computes_and_writes_every_tier_for_a_diacritic_preserving_model(self, tmp_path):
        calls = []

        def _fake_compute(psalms, model_name, *, vocalized, niqqud_only, device, torch_dtype):
            calls.append((model_name, vocalized, niqqud_only))
            return {p.number: np.zeros((len(p.half_verses), 2)) for p in psalms}

        psalms = [_psalm(number=1, half_verses=("A",), half_verses_unvocalized=("a",))]

        written = generate_local(psalms, tmp_path, "bge-m3", compute=_fake_compute)

        assert written == [
            "semantic_bge_m3_consonantal",
            "semantic_bge_m3_vocalized",
            "semantic_bge_m3_cantillation",
        ]
        assert len(calls) == 3
        for name in written:
            assert feature_path(tmp_path, name).exists()

    def test_computes_only_one_tier_for_a_diacritic_stripping_model(self, tmp_path):
        calls = []

        def _fake_compute(psalms, model_name, *, vocalized, niqqud_only, device, torch_dtype):
            calls.append(model_name)
            return {p.number: np.zeros((len(p.half_verses), 2)) for p in psalms}

        psalms = [_psalm(number=1, half_verses=("A",), half_verses_unvocalized=("a",))]

        written = generate_local(psalms, tmp_path, "miqrabert", compute=_fake_compute)

        assert written == ["semantic_miqrabert_consonantal"]
        assert len(calls) == 1

    def test_skips_a_tier_whose_tf_file_already_exists(self, tmp_path):
        from tehillim_embeddings.export import node_values, write_feature

        psalms = [_psalm(number=1, half_verses=("A",), half_verses_unvocalized=("a",))]
        write_feature(
            tmp_path,
            "semantic_miqrabert_consonantal",
            node_values({1: np.zeros((1, 2))}, psalms),
            "already here",
        )

        def _must_not_be_called(*args, **kwargs):
            raise AssertionError("compute must not be called for an already-written tier")

        written = generate_local(psalms, tmp_path, "miqrabert", compute=_must_not_be_called)

        assert written == []

    def test_passes_device_and_torch_dtype_through(self, tmp_path):
        calls = []

        def _fake_compute(psalms, model_name, *, vocalized, niqqud_only, device, torch_dtype):
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
    def test_cache_miss_calls_fetch_once_per_tier_with_flattened_half_verses(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("TEHILLIM_OPENROUTER_API_KEY", "test-key")
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

        written = generate_api(psalms, tmp_path, "gemini", fetch=_fake_fetch)

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
        from tehillim_embeddings.export import node_values, write_feature

        psalms = [_psalm(number=1, half_verses=("A",), half_verses_unvocalized=("a",))]
        for name in [
            "semantic_gemini_embedding_2_consonantal",
            "semantic_gemini_embedding_2_vocalized",
            "semantic_gemini_embedding_2_cantillation",
        ]:
            write_feature(
                tmp_path, name, node_values({1: np.zeros((1, 3))}, psalms), "already here"
            )

        def _must_not_be_called(texts, *, api_key):
            raise AssertionError("fetch must not be called on a cache hit")

        written = generate_api(psalms, tmp_path, "gemini", fetch=_must_not_be_called)

        assert written == []

    def test_missing_api_key_raises_naming_the_exact_env_var(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TEHILLIM_OPENROUTER_API_KEY", raising=False)
        psalms = [_psalm(number=1, half_verses=("A",), half_verses_unvocalized=("a",))]

        def _fake_fetch(texts, *, api_key):
            return np.zeros((1, 1))

        with pytest.raises(RuntimeError, match="TEHILLIM_OPENROUTER_API_KEY"):
            generate_api(psalms, tmp_path, "gemini", fetch=_fake_fetch)

    def test_missing_api_key_is_not_read_when_every_tier_is_already_cached(
        self, tmp_path, monkeypatch
    ):
        from tehillim_embeddings.export import node_values, write_feature

        monkeypatch.delenv("TEHILLIM_OPENROUTER_API_KEY", raising=False)
        psalms = [_psalm(number=1, half_verses=("A",), half_verses_unvocalized=("a",))]
        for name in [
            "semantic_gemini_embedding_2_consonantal",
            "semantic_gemini_embedding_2_vocalized",
            "semantic_gemini_embedding_2_cantillation",
        ]:
            write_feature(
                tmp_path, name, node_values({1: np.zeros((1, 3))}, psalms), "already here"
            )

        written = generate_api(
            psalms, tmp_path, "gemini", fetch=lambda texts, *, api_key: np.zeros((1, 1))
        )

        assert written == []

    def test_cohere_uses_its_own_api_key_env_var(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TEHILLIM_OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("TEHILLIM_COHERE_API_KEY", "cohere-key")
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

        generate_api(psalms, tmp_path, "cohere", fetch=_fake_fetch)

        assert calls and all(key == "cohere-key" for key in calls)
