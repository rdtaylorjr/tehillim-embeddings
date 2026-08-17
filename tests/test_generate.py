"""Unit tests for generate.py's orchestration: which (model, variation)
pairs get computed, the .parquet-file-exists cache check, and API key
handling.
"""

from __future__ import annotations

import numpy as np
import pytest

from semantic.export import dataset_path
from semantic.generate import generate_api, generate_local


def _psalm(*, number: int, half_verses, half_verses_unvocalized, half_verses_niqqud_only=()):
    from semantic.corpus import Psalm

    return Psalm(
        number=number,
        half_verses=half_verses,
        half_verses_unvocalized=half_verses_unvocalized,
        half_verses_niqqud_only=half_verses_niqqud_only,
        half_verse_nodes=tuple(range(number * 100, number * 100 + len(half_verses))),
    )


class TestGenerateLocal:
    def test_computes_and_writes_every_variation_for_a_diacritic_preserving_model(self, tmp_path):
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
        for variation in ("consonantal", "vocalized", "cantillation"):
            assert dataset_path(tmp_path, "bge_m3", variation).exists()

    def test_computes_only_one_variation_for_a_diacritic_stripping_model(self, tmp_path):
        calls = []

        def _fake_compute(psalms, model_name, *, vocalized, niqqud_only, device, torch_dtype):
            calls.append((model_name, vocalized, niqqud_only))
            return {p.number: np.zeros((len(p.half_verses), 2)) for p in psalms}

        psalms = [_psalm(number=1, half_verses=("A",), half_verses_unvocalized=("a",))]

        written = generate_local(psalms, tmp_path, "miqrabert", compute=_fake_compute)

        assert written == ["semantic_miqrabert_consonantal"]
        assert calls == [("davidmsmiley/MiqraBERT", False, False)]

    def test_restricts_to_a_single_named_variation(self, tmp_path):
        calls = []

        def _fake_compute(psalms, model_name, *, vocalized, niqqud_only, device, torch_dtype):
            calls.append((vocalized, niqqud_only))
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

        assert calls and all(key == "cohere-key" for key in calls)
