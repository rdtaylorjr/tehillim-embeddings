"""Opens a local embedding model as an Encoder: one call over texts, one row each."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator, MutableMapping, Sequence
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from semantic.units import Encoder

if TYPE_CHECKING:
    import torch

MIQRABERT_MODEL = "davidmsmiley/MiqraBERT"
ALEPHBERT_MODEL = "imvladikon/sentence-transformers-alephbert"
NEODICTABERT_MODEL = "dicta-il/neodictabert-bilingual-embed"
BEREL_MODEL = "dicta-il/BEREL"
BGE_MULTILINGUAL_GEMMA2_MODEL = "BAAI/bge-multilingual-gemma2"
QWEN3_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
KALM_EMBEDDING_MODEL = "tencent/KaLM-Embedding-Gemma3-12B-2511"
LLAMA_EMBED_NEMOTRON_MODEL = "nvidia/llama-embed-nemotron-8b"
HARRIER_OSS_MODEL = "microsoft/harrier-oss-v1-27b"
F2LLM_V2_MODEL = "codefuse-ai/F2LLM-v2-14B"
BGE_M3_MODEL = "BAAI/bge-m3"
GTE_MULTILINGUAL_MODEL = "Alibaba-NLP/gte-multilingual-base"

#: This checkpoint's NaN failure is intermittent even on CPU, so retries with eviction are the fix.
_GTE_MULTILINGUAL_MAX_ATTEMPTS = 5

ME5_LARGE_INSTRUCT_MODEL = "intfloat/multilingual-e5-large-instruct"


def _rope_frequencies(
    dim: int, length: int, theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reimplements NeoDictaBERT's `precompute_freqs` deterministically."""
    import torch

    half = dim // 2
    idx = torch.arange(0, half, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** ((2.0 * idx) / dim))
    positions = torch.arange(length, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    return angles.cos(), angles.sin()


def _repair_neodictabert_rope_buffers(auto_model: Any) -> None:
    """Recomputes RoPE buffers: persistent=False means from_pretrained never fills them."""
    for module in auto_model.modules():
        freqs_cos = getattr(module, "freqs_cos", None)
        if freqs_cos is None:
            continue
        dim = module.config.hidden_size // module.config.num_attention_heads
        cos, sin = _rope_frequencies(dim, module.config.max_length)
        #: copy_ would broadcast a mismatched recomputation into the buffer silently.
        if tuple(freqs_cos.shape) != tuple(cos.shape):
            raise RuntimeError(
                f"NeoDictaBERT RoPE buffer shape {tuple(freqs_cos.shape)} does not match the "
                f"recomputed {tuple(cos.shape)}, refusing to overwrite it"
            )
        module.freqs_cos.copy_(cos)
        module.freqs_sin.copy_(sin)


def _evict_gte_multilingual_dynamic_module(
    modules: MutableMapping[str, Any] | None = None,
) -> None:
    """Evicts this checkpoint's cached dynamic module so a retry reimports it."""
    if modules is None:
        modules = sys.modules
    stale = [name for name in modules if "transformers_modules.Alibaba_hyphen_NLP" in name]
    for name in stale:
        del modules[name]


def _repair_gte_multilingual_position_ids(auto_model: Any) -> None:
    """Overwrites this checkpoint's `position_ids` buffer if it loaded as garbage."""
    import torch

    for module in auto_model.modules():
        position_ids = getattr(module, "position_ids", None)
        if position_ids is None or not torch.is_tensor(position_ids) or position_ids.ndim == 0:
            continue
        #: Buffers are (max_positions,) or (1, max_positions), so the ramp runs on the last axis.
        expected = torch.arange(position_ids.shape[-1], device=position_ids.device).expand_as(
            position_ids
        )
        if torch.equal(position_ids, expected):
            continue
        module.position_ids.copy_(expected)


def _extended_attention_mask(
    attention_mask: torch.Tensor, input_shape: Any, device: Any = None, dtype: Any = None
) -> torch.Tensor:
    """Transformers 4.x's `PreTrainedModel.get_extended_attention_mask`, which 5.x removed."""
    import torch

    del input_shape, device
    dtype = torch.float32 if dtype is None else dtype
    if attention_mask.dim() == 3:
        extended = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended = attention_mask[:, None, None, :]
    else:
        raise ValueError(f"attention mask has {attention_mask.dim()} dimensions, expected 2 or 3")
    extended = extended.to(dtype=dtype)
    return (1.0 - extended) * torch.finfo(dtype).min


def _repair_gte_multilingual_attention_mask(auto_model: Any) -> None:
    """Restores the 4.x mask helper this checkpoint's remote code calls, where 5.x lacks it."""
    if not hasattr(auto_model, "get_extended_attention_mask"):
        auto_model.get_extended_attention_mask = partial(
            _extended_attention_mask, dtype=auto_model.dtype
        )


#: SentenceTransformer can't load these: KaLM crashes, Harrier needs a processor it lacks.
RAW_TRANSFORMER_MODELS = {KALM_EMBEDDING_MODEL, HARRIER_OSS_MODEL}

#: Models whose own documented usage never passes trust_remote_code=True.
NO_TRUST_REMOTE_CODE_MODELS = {HARRIER_OSS_MODEL, F2LLM_V2_MODEL}


def _load_raw_transformer(
    model_name: str,
    *,
    torch_dtype: str | None,
    device: str | None,
    load_tokenizer: Callable[..., Any] | None = None,
    load_model: Callable[..., Any] | None = None,
    cuda_available: Callable[[], bool] | None = None,
    mps_available: Callable[[], bool] | None = None,
) -> tuple[Any, Any]:
    """Loads a RAW_TRANSFORMER_MODELS checkpoint via plain transformers, bypassing ST."""
    if load_tokenizer is None or load_model is None:
        from transformers import AutoModel, AutoTokenizer

        if load_tokenizer is None:
            load_tokenizer = AutoTokenizer.from_pretrained
        if load_model is None:
            load_model = AutoModel.from_pretrained
    if cuda_available is None or mps_available is None:
        import torch

        if cuda_available is None:
            cuda_available = torch.cuda.is_available
        if mps_available is None:
            mps_available = torch.backends.mps.is_available

    trust_remote_code = model_name not in NO_TRUST_REMOTE_CODE_MODELS
    tokenizer = load_tokenizer(model_name, trust_remote_code=trust_remote_code)
    model_kwargs = {"torch_dtype": torch_dtype} if torch_dtype is not None else {}
    model = load_model(model_name, trust_remote_code=trust_remote_code, **model_kwargs)
    if device is not None:
        resolved_device = device
    elif cuda_available():
        resolved_device = "cuda"
    elif mps_available():
        resolved_device = "mps"
    else:
        resolved_device = "cpu"
    model = model.to(resolved_device)
    model.eval()
    return tokenizer, model


def _encode_last_token_pooled(texts: list[str], tokenizer: Any, model: Any) -> np.ndarray:
    """Last-token pooling and L2 normalization, correct under either padding side."""
    import torch

    device = next(model.parameters()).device
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]
    left_padded = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padded:
        pooled = hidden[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.shape[0], device=device), sequence_lengths]
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled.to(torch.float32).cpu().numpy()


def _release_gpu_memory() -> None:
    """Frees cached GPU memory after a model is deleted."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


#: Texts per forward pass of a raw transformer, whose pooler holds one batch on the device.
RAW_BATCH = 32


def _in_batches(
    encode: Callable[[list[str]], np.ndarray], batch: int
) -> Callable[[Sequence[str]], np.ndarray]:
    """Runs an encoder over slices of a text list and stacks the rows."""

    def encode_all(texts: Sequence[str]) -> np.ndarray:
        rows = [encode(list(texts[start : start + batch])) for start in range(0, len(texts), batch)]
        return np.concatenate(rows, axis=0)

    return encode_all


@contextmanager
def _raw_transformer_encoder(
    model_name: str,
    *,
    device: str | None,
    torch_dtype: str | None,
    raw_transformer_loader: Callable[..., tuple[Any, Any]],
    last_token_pooler: Callable[[list[str], Any, Any], np.ndarray],
    release_gpu_memory: Callable[[], None],
) -> Iterator[Encoder]:
    """Plain transformers with last-token pooling, for checkpoints ST cannot load."""
    loaded: list[Any] = list(
        raw_transformer_loader(model_name, torch_dtype=torch_dtype, device=device)
    )

    def pooled(texts: list[str]) -> np.ndarray:
        return last_token_pooler(texts, loaded[0], loaded[1])

    try:
        yield _in_batches(pooled, RAW_BATCH)
    finally:
        loaded.clear()
        release_gpu_memory()


@contextmanager
def _sentence_transformer_encoder(
    model_name: str,
    *,
    device: str | None,
    torch_dtype: str | None,
    factory: Callable[..., Any],
    release_gpu_memory: Callable[[], None],
) -> Iterator[Encoder]:
    """A sentence-transformers model with the repairs its checkpoint needs, reloaded on NaNs."""
    if model_name == GTE_MULTILINGUAL_MODEL:
        #: transformers 5 loads the checkpoint's float16 weights as they are, which overflow on
        #: the CPU, where 4.x upcast them to float32.
        device = "cpu" if device is None else device
        torch_dtype = "float32" if torch_dtype is None else torch_dtype
    model_kwargs = {"model_kwargs": {"torch_dtype": torch_dtype}} if torch_dtype is not None else {}
    trust_remote_code = model_name not in NO_TRUST_REMOTE_CODE_MODELS

    def load() -> Any:
        model = factory(
            model_name, trust_remote_code=trust_remote_code, device=device, **model_kwargs
        )
        if model_name == NEODICTABERT_MODEL:
            _repair_neodictabert_rope_buffers(model[0].auto_model)
        if model_name == GTE_MULTILINGUAL_MODEL:
            _repair_gte_multilingual_position_ids(model[0].auto_model)
            _repair_gte_multilingual_attention_mask(model[0].auto_model)
        return model

    loaded: list[Any] = [load()]
    #: This checkpoint's NaN failure is intermittent, so a non-finite result reloads it and retries.
    attempts = _GTE_MULTILINGUAL_MAX_ATTEMPTS if model_name == GTE_MULTILINGUAL_MODEL else 1

    def encode(texts: Sequence[str]) -> np.ndarray:
        for attempt in range(1, attempts + 1):
            vectors = np.asarray(loaded[0].encode(list(texts), normalize_embeddings=True))
            if np.all(np.isfinite(vectors)):
                return vectors
            print(
                f"warning: {model_name} produced non-finite embeddings on attempt "
                f"{attempt}/{attempts}, retrying with a fresh model load",
                file=sys.stderr,
            )
            if attempt < attempts:
                loaded.clear()
                release_gpu_memory()
                _evict_gte_multilingual_dynamic_module()
                loaded.append(load())
        raise RuntimeError(
            f"{model_name} produced non-finite embeddings on every one of {attempts} attempts"
        )

    try:
        yield encode
    finally:
        loaded.clear()
        release_gpu_memory()


@contextmanager
def encoder(
    model_name: str,
    *,
    device: str | None = None,
    torch_dtype: str | None = None,
    sentence_transformer_factory: Callable[..., Any] | None = None,
    raw_transformer_loader: Callable[..., tuple[Any, Any]] = _load_raw_transformer,
    last_token_pooler: Callable[[list[str], Any, Any], np.ndarray] = _encode_last_token_pooled,
    release_gpu_memory: Callable[[], None] = _release_gpu_memory,
) -> Iterator[Encoder]:
    """Loads one model on the best device available, yields it as an Encoder, then frees it."""
    if model_name in RAW_TRANSFORMER_MODELS:
        opened = _raw_transformer_encoder(
            model_name,
            device=device,
            torch_dtype=torch_dtype,
            raw_transformer_loader=raw_transformer_loader,
            last_token_pooler=last_token_pooler,
            release_gpu_memory=release_gpu_memory,
        )
    else:
        if sentence_transformer_factory is None:
            from sentence_transformers import SentenceTransformer

            factory = cast("Callable[..., Any]", SentenceTransformer)
        else:
            factory = sentence_transformer_factory
        opened = _sentence_transformer_encoder(
            model_name,
            device=device,
            torch_dtype=torch_dtype,
            factory=factory,
            release_gpu_memory=release_gpu_memory,
        )
    with opened as encode:
        yield encode
