"""Unit tests for local_models.py. Every seam (model loader, encoder,
device probe, sys.modules) is passed as an explicit function argument,
defaulting to the real implementation in production and to a fake in
these tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from semantic.local_models import (
    KALM_EMBEDDING_MODEL,
    _encode_last_token_pooled,
    _evict_gte_multilingual_dynamic_module,
    _load_kalm_raw_transformer,
    _repair_gte_multilingual_position_ids,
    _repair_neodictabert_rope_buffers,
    _rope_frequencies,
    _select_half_verses,
    compute_half_verse_embeddings,
)


def _psalm(
    *,
    number: int = 1,
    half_verses: tuple[str, ...],
    half_verses_unvocalized: tuple[str, ...],
    half_verses_niqqud_only: tuple[str, ...] = (),
):
    from semantic.corpus import Psalm

    return Psalm(
        number=number,
        half_verses=half_verses,
        half_verses_unvocalized=half_verses_unvocalized,
        half_verses_niqqud_only=half_verses_niqqud_only,
    )


class TestSelectHalfVerses:
    def test_vocalized_true_selects_half_verses(self):
        psalm = _psalm(half_verses=("A",), half_verses_unvocalized=("B",))
        assert _select_half_verses(psalm, vocalized=True) == ("A",)

    def test_vocalized_false_selects_half_verses_unvocalized(self):
        psalm = _psalm(half_verses=("A",), half_verses_unvocalized=("B",))
        assert _select_half_verses(psalm, vocalized=False) == ("B",)

    def test_niqqud_only_overrides_vocalized_entirely(self):
        psalm = _psalm(
            half_verses=("A",),
            half_verses_unvocalized=("B",),
            half_verses_niqqud_only=("C",),
        )
        assert _select_half_verses(psalm, vocalized=True, niqqud_only=True) == ("C",)
        assert _select_half_verses(psalm, vocalized=False, niqqud_only=True) == ("C",)


class TestRopeFrequencies:
    def test_is_deterministic(self):
        cos1, sin1 = _rope_frequencies(dim=8, length=4)
        cos2, sin2 = _rope_frequencies(dim=8, length=4)
        assert np.array_equal(cos1.numpy(), cos2.numpy())
        assert np.array_equal(sin1.numpy(), sin2.numpy())

    def test_produces_no_nan(self):
        cos, sin = _rope_frequencies(dim=64, length=1024)
        assert not np.isnan(cos.numpy()).any()
        assert not np.isnan(sin.numpy()).any()

    def test_shape_is_length_by_half_dim(self):
        cos, sin = _rope_frequencies(dim=8, length=4)
        assert tuple(cos.shape) == (4, 4)
        assert tuple(sin.shape) == (4, 4)

    def test_first_position_is_the_identity_angle(self):
        cos, sin = _rope_frequencies(dim=8, length=4)
        assert np.allclose(cos.numpy()[0], 1.0)
        assert np.allclose(sin.numpy()[0], 0.0)


class TestRepairNeodictabertRopeBuffers:
    class _StubConfig:
        def __init__(self, hidden_size=8, num_attention_heads=2, max_length=4):
            self.hidden_size = hidden_size
            self.num_attention_heads = num_attention_heads
            self.max_length = max_length

    class _StubRopeModule:
        def __init__(self, freqs_cos, freqs_sin, config):
            self.freqs_cos = freqs_cos
            self.freqs_sin = freqs_sin
            self.config = config

    class _StubAutoModel:
        def __init__(self, submodules):
            self._submodules = submodules

        def modules(self):
            return iter(self._submodules)

    def test_repairs_a_nan_buffer_in_place(self):
        import torch

        config = self._StubConfig()
        broken = self._StubRopeModule(
            freqs_cos=torch.full((4, 2), float("nan")),
            freqs_sin=torch.full((4, 2), float("nan")),
            config=config,
        )
        auto_model = self._StubAutoModel([broken])

        _repair_neodictabert_rope_buffers(auto_model)

        assert not torch.isnan(broken.freqs_cos).any()
        assert not torch.isnan(broken.freqs_sin).any()
        expected_cos, expected_sin = _rope_frequencies(dim=4, length=4)
        assert torch.equal(broken.freqs_cos, expected_cos)
        assert torch.equal(broken.freqs_sin, expected_sin)

    def test_leaves_a_healthy_buffer_untouched(self):
        import torch

        config = self._StubConfig()
        cos, sin = _rope_frequencies(dim=4, length=4)
        healthy = self._StubRopeModule(freqs_cos=cos.clone(), freqs_sin=sin.clone(), config=config)
        auto_model = self._StubAutoModel([healthy])

        _repair_neodictabert_rope_buffers(auto_model)

        assert torch.equal(healthy.freqs_cos, cos)
        assert torch.equal(healthy.freqs_sin, sin)

    def test_ignores_modules_without_freqs_cos(self):
        class _Unrelated:
            pass

        auto_model = self._StubAutoModel([_Unrelated()])

        _repair_neodictabert_rope_buffers(auto_model)  # must not raise


class TestRepairGteMultilingualPositionIds:
    class _StubPositionIdsModule:
        def __init__(self, position_ids):
            self.position_ids = position_ids

    class _StubAutoModel:
        def __init__(self, submodules):
            self._submodules = submodules

        def modules(self):
            return iter(self._submodules)

    def test_repairs_a_garbage_buffer_in_place(self):
        import torch

        broken = self._StubPositionIdsModule(
            position_ids=torch.tensor([0, 4335441888, 59023, -1, 7453010313431162915])
        )
        auto_model = self._StubAutoModel([broken])

        _repair_gte_multilingual_position_ids(auto_model)

        assert torch.equal(broken.position_ids, torch.arange(5))

    def test_leaves_a_healthy_buffer_untouched(self):
        import torch

        healthy = self._StubPositionIdsModule(position_ids=torch.arange(8192))
        auto_model = self._StubAutoModel([healthy])

        _repair_gte_multilingual_position_ids(auto_model)

        assert torch.equal(healthy.position_ids, torch.arange(8192))

    def test_ignores_modules_without_position_ids(self):
        class _Unrelated:
            pass

        auto_model = self._StubAutoModel([_Unrelated()])

        _repair_gte_multilingual_position_ids(auto_model)  # must not raise

    def test_ignores_non_tensor_position_ids_attribute(self):
        class _WeirdModule:
            position_ids = None

        auto_model = self._StubAutoModel([_WeirdModule()])

        _repair_gte_multilingual_position_ids(auto_model)  # must not raise


class TestEvictGteMultilingualDynamicModule:
    def test_purges_matching_dynamic_module_entries(self):
        fake_modules = {
            "transformers_modules.Alibaba_hyphen_NLP.new_hyphen_impl.abc123.modeling": (
                object()
            ),
            "transformers_modules.Alibaba_hyphen_NLP.new_hyphen_impl.abc123.configuration": (
                object()
            ),
            "some_other_module": object(),
        }

        _evict_gte_multilingual_dynamic_module(fake_modules)

        assert not any("transformers_modules.Alibaba_hyphen_NLP" in name for name in fake_modules)
        assert "some_other_module" in fake_modules

    def test_no_op_when_nothing_matches(self):
        fake_modules = {"some_other_module": object()}

        _evict_gte_multilingual_dynamic_module(fake_modules)  # must not raise

        assert "some_other_module" in fake_modules


class TestEncodeLastTokenPooled:
    def test_picks_the_last_non_padded_token_not_the_first_or_mean(self):
        import torch

        class _FakeTokenizer:
            def __call__(self, texts, *, padding, truncation, max_length, return_tensors):
                return {
                    "input_ids": torch.zeros((2, 3), dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
                }

        class _FakeModel:
            class _Output:
                def __init__(self, hidden):
                    self.last_hidden_state = hidden

            def __call__(self, **inputs):
                hidden = torch.tensor(
                    [
                        [[1.0, 0.0], [0.0, 1.0], [3.0, 4.0]],
                        [[5.0, 0.0], [0.0, 5.0], [9.0, 9.0]],
                    ]
                )
                return self._Output(hidden)

            def parameters(self):
                return iter([torch.zeros(1)])

        result = _encode_last_token_pooled(["a", "b"], _FakeTokenizer(), _FakeModel())

        expected = torch.nn.functional.normalize(
            torch.tensor([[3.0, 4.0], [0.0, 5.0]]), p=2, dim=1
        ).numpy()
        assert np.allclose(result, expected)

    def test_output_rows_are_unit_norm(self):
        import torch

        class _FakeTokenizer:
            def __call__(self, texts, *, padding, truncation, max_length, return_tensors):
                return {
                    "input_ids": torch.zeros((1, 2), dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1]]),
                }

        class _FakeModel:
            class _Output:
                def __init__(self, hidden):
                    self.last_hidden_state = hidden

            def __call__(self, **inputs):
                return self._Output(torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]))

            def parameters(self):
                return iter([torch.zeros(1)])

        result = _encode_last_token_pooled(["a"], _FakeTokenizer(), _FakeModel())

        assert np.allclose(np.linalg.norm(result, axis=1), 1.0)


class TestLoadKalmRawTransformerDevice:
    class _FakeModel:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

    def _fake_loaders(self):
        return (
            lambda model_name, *, trust_remote_code: "fake-tokenizer",
            lambda model_name, *, trust_remote_code, **kwargs: self._FakeModel(),
        )

    def test_explicit_device_is_used_as_given_even_if_cuda_available(self):
        load_tokenizer, load_model = self._fake_loaders()

        _, model = _load_kalm_raw_transformer(
            "some-model",
            torch_dtype=None,
            device="cpu",
            load_tokenizer=load_tokenizer,
            load_model=load_model,
            cuda_available=lambda: True,
            mps_available=lambda: True,
        )

        assert model.device == "cpu"

    def test_none_device_prefers_cuda_when_available(self):
        load_tokenizer, load_model = self._fake_loaders()

        _, model = _load_kalm_raw_transformer(
            "some-model",
            torch_dtype=None,
            device=None,
            load_tokenizer=load_tokenizer,
            load_model=load_model,
            cuda_available=lambda: True,
            mps_available=lambda: True,
        )

        assert model.device == "cuda"

    def test_none_device_falls_back_to_mps_when_cuda_unavailable(self):
        load_tokenizer, load_model = self._fake_loaders()

        _, model = _load_kalm_raw_transformer(
            "some-model",
            torch_dtype=None,
            device=None,
            load_tokenizer=load_tokenizer,
            load_model=load_model,
            cuda_available=lambda: False,
            mps_available=lambda: True,
        )

        assert model.device == "mps"

    def test_none_device_falls_back_to_cpu_when_neither_available(self):
        load_tokenizer, load_model = self._fake_loaders()

        _, model = _load_kalm_raw_transformer(
            "some-model",
            torch_dtype=None,
            device=None,
            load_tokenizer=load_tokenizer,
            load_model=load_model,
            cuda_available=lambda: False,
            mps_available=lambda: False,
        )

        assert model.device == "cpu"


class TestComputeHalfVerseEmbeddingsDevice:
    def test_passes_device_through_to_sentence_transformer_factory(self):
        calls = []

        class _FakeModel:
            def __getitem__(self, index):
                raise AssertionError("only NeoDictaBERT/GTE call model[0]")

            def encode(self, texts, normalize_embeddings=True):
                return np.zeros((len(texts), 3))

        def _fake_factory(model_name, *, trust_remote_code=False, device=None):
            calls.append(
                {
                    "model_name": model_name,
                    "trust_remote_code": trust_remote_code,
                    "device": device,
                }
            )
            return _FakeModel()

        psalm = _psalm(half_verses=("A",), half_verses_unvocalized=("B",))

        compute_half_verse_embeddings(
            [psalm], "some-model", device="cpu", sentence_transformer_factory=_fake_factory
        )

        assert calls == [{"model_name": "some-model", "trust_remote_code": True, "device": "cpu"}]

    def test_device_defaults_to_none(self):
        calls = []

        class _FakeModel:
            def encode(self, texts, normalize_embeddings=True):
                return np.zeros((len(texts), 3))

        def _fake_factory(model_name, *, trust_remote_code=False, device=None):
            calls.append(device)
            return _FakeModel()

        psalm = _psalm(half_verses=("A",), half_verses_unvocalized=("B",))

        compute_half_verse_embeddings(
            [psalm], "some-model", sentence_transformer_factory=_fake_factory
        )

        assert calls == [None]


class TestComputeHalfVerseEmbeddingsTorchDtype:
    def test_passes_torch_dtype_through_as_model_kwargs(self):
        calls = []

        class _FakeModel:
            def encode(self, texts, normalize_embeddings=True):
                return np.zeros((len(texts), 3))

        def _fake_factory(model_name, *, trust_remote_code=False, device=None, model_kwargs=None):
            calls.append(model_kwargs)
            return _FakeModel()

        psalm = _psalm(half_verses=("A",), half_verses_unvocalized=("B",))

        compute_half_verse_embeddings(
            [psalm], "some-model", torch_dtype="float16", sentence_transformer_factory=_fake_factory
        )

        assert calls == [{"torch_dtype": "float16"}]

    def test_omits_model_kwargs_entirely_when_torch_dtype_is_none(self):
        calls = []

        class _FakeModel:
            def encode(self, texts, normalize_embeddings=True):
                return np.zeros((len(texts), 3))

        def _fake_factory(model_name, *, trust_remote_code=False, device=None):
            # Deliberately does not accept model_kwargs. If
            # compute_half_verse_embeddings passed it when torch_dtype
            # isn't given, this fake would raise TypeError.
            calls.append(True)
            return _FakeModel()

        psalm = _psalm(half_verses=("A",), half_verses_unvocalized=("B",))

        compute_half_verse_embeddings(
            [psalm], "some-model", sentence_transformer_factory=_fake_factory
        )

        assert calls == [True]


class TestComputeHalfVerseEmbeddingsKalmRoutesAroundSentenceTransformer:
    def test_kalm_model_name_never_touches_sentence_transformer_factory(self):
        def _fail(*args, **kwargs):
            raise AssertionError("sentence_transformer_factory must not be used for KaLM")

        calls = []

        def _fake_kalm_loader(model_name, *, torch_dtype, device):
            calls.append((model_name, torch_dtype, device))
            return "fake-tokenizer", "fake-model"

        def _fake_pooler(texts, tokenizer, model):
            assert tokenizer == "fake-tokenizer"
            assert model == "fake-model"
            return np.zeros((len(texts), 4))

        psalm = _psalm(half_verses=("A", "B"), half_verses_unvocalized=("C", "D"))

        result = compute_half_verse_embeddings(
            [psalm],
            KALM_EMBEDDING_MODEL,
            torch_dtype="bfloat16",
            device="cuda",
            sentence_transformer_factory=_fail,
            kalm_loader=_fake_kalm_loader,
            last_token_pooler=_fake_pooler,
            release_gpu_memory=lambda: None,
        )

        assert calls == [(KALM_EMBEDDING_MODEL, "bfloat16", "cuda")]
        assert result[1].shape == (2, 4)

    def test_loads_once_and_reuses_across_psalms(self):
        load_calls = []
        encode_calls = []

        def _fake_kalm_loader(model_name, *, torch_dtype, device):
            load_calls.append(model_name)
            return "tok", "mdl"

        def _fake_pooler(texts, tokenizer, model):
            encode_calls.append(texts)
            return np.zeros((len(texts), 2))

        psalms = [
            _psalm(number=1, half_verses=("A",), half_verses_unvocalized=("a",)),
            _psalm(number=2, half_verses=("B",), half_verses_unvocalized=("b",)),
        ]

        compute_half_verse_embeddings(
            psalms,
            KALM_EMBEDDING_MODEL,
            kalm_loader=_fake_kalm_loader,
            last_token_pooler=_fake_pooler,
            release_gpu_memory=lambda: None,
        )

        assert len(load_calls) == 1
        assert len(encode_calls) == 2


class TestComputeHalfVerseEmbeddingsGteRetry:
    # compute_half_verse_embeddings calls _repair_gte_multilingual_
    # position_ids(model[0].auto_model) for this model name, so the fake
    # must expose that same shape (a real SentenceTransformer's module
    # list, subscriptable, with .auto_model on element 0).
    class _FakeAutoModel:
        def modules(self):
            return iter(())

    class _FakeSubModule:
        def __init__(self):
            self.auto_model = TestComputeHalfVerseEmbeddingsGteRetry._FakeAutoModel()

    class _FakeModel:
        def __init__(self, ok):
            self._ok = ok
            self._sub_modules = [TestComputeHalfVerseEmbeddingsGteRetry._FakeSubModule()]

        def __getitem__(self, index):
            return self._sub_modules[index]

        def encode(self, texts, normalize_embeddings=True):
            value = 1.0 if self._ok else float("nan")
            return np.full((len(texts), 3), value)

    def test_retries_with_a_fresh_model_on_non_finite_output(self):
        from semantic.local_models import GTE_MULTILINGUAL_MODEL

        attempts = []

        def _fake_factory(model_name, *, trust_remote_code=False, device=None, model_kwargs=None):
            attempts.append(1)
            return self._FakeModel(ok=len(attempts) >= 2)

        psalm = _psalm(half_verses=("A",), half_verses_unvocalized=("B",))

        result = compute_half_verse_embeddings(
            [psalm],
            GTE_MULTILINGUAL_MODEL,
            sentence_transformer_factory=_fake_factory,
            release_gpu_memory=lambda: None,
        )

        assert len(attempts) == 2
        assert np.all(np.isfinite(result[1]))

    def test_raises_after_every_attempt_stays_non_finite(self):
        from semantic.local_models import GTE_MULTILINGUAL_MODEL

        def _fake_factory(model_name, *, trust_remote_code=False, device=None, model_kwargs=None):
            return self._FakeModel(ok=False)

        psalm = _psalm(half_verses=("A",), half_verses_unvocalized=("B",))

        with pytest.raises(RuntimeError, match="non-finite"):
            compute_half_verse_embeddings(
                [psalm],
                GTE_MULTILINGUAL_MODEL,
                sentence_transformer_factory=_fake_factory,
                release_gpu_memory=lambda: None,
            )


class TestComputeHalfVerseEmbeddingsRepairsAreWired:
    # compute_half_verse_embeddings calls the repair functions directly
    # on model[0].auto_model for these two model names. These tests
    # confirm that call actually happens, not just that the repair
    # functions work correctly in isolation.
    class _FakeSubModule:
        def __init__(self, auto_model):
            self.auto_model = auto_model

    class _FakeModel:
        def __init__(self, auto_model):
            sub_module_cls = TestComputeHalfVerseEmbeddingsRepairsAreWired._FakeSubModule
            self._sub_modules = [sub_module_cls(auto_model)]

        def __getitem__(self, index):
            return self._sub_modules[index]

        def encode(self, texts, normalize_embeddings=True):
            return np.zeros((len(texts), 3))

    def test_neodictabert_rope_buffer_is_repaired(self):
        import torch

        from semantic.local_models import NEODICTABERT_MODEL

        class _StubConfig:
            hidden_size = 8
            num_attention_heads = 2
            max_length = 4

        class _StubRopeModule:
            def __init__(self):
                self.freqs_cos = torch.full((4, 2), float("nan"))
                self.freqs_sin = torch.full((4, 2), float("nan"))
                self.config = _StubConfig()

        class _StubAutoModel:
            def __init__(self, submodule):
                self._submodule = submodule

            def modules(self):
                return iter([self._submodule])

        broken = _StubRopeModule()
        fake_model = self._FakeModel(_StubAutoModel(broken))

        def _fake_factory(model_name, *, trust_remote_code=False, device=None):
            return fake_model

        psalm = _psalm(half_verses=("A",), half_verses_unvocalized=("B",))

        compute_half_verse_embeddings(
            [psalm],
            NEODICTABERT_MODEL,
            sentence_transformer_factory=_fake_factory,
            release_gpu_memory=lambda: None,
        )

        assert not torch.isnan(broken.freqs_cos).any()

    def test_gte_multilingual_position_ids_buffer_is_repaired(self):
        import torch

        from semantic.local_models import GTE_MULTILINGUAL_MODEL

        class _StubPositionIdsModule:
            def __init__(self):
                self.position_ids = torch.tensor([0, 4335441888, 59023, -1, 7453010313431162915])

        class _StubAutoModel:
            def __init__(self, submodule):
                self._submodule = submodule

            def modules(self):
                return iter([self._submodule])

        broken = _StubPositionIdsModule()
        fake_model = self._FakeModel(_StubAutoModel(broken))

        def _fake_factory(model_name, *, trust_remote_code=False, device=None, model_kwargs=None):
            return fake_model

        psalm = _psalm(half_verses=("A",), half_verses_unvocalized=("B",))

        compute_half_verse_embeddings(
            [psalm],
            GTE_MULTILINGUAL_MODEL,
            sentence_transformer_factory=_fake_factory,
            release_gpu_memory=lambda: None,
        )

        assert torch.equal(broken.position_ids, torch.arange(5))
