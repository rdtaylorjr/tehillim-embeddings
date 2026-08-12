"""Computes half-verse embeddings from local sentence-embedding models."""

from __future__ import annotations

import sys
from collections.abc import Callable, MutableMapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch
    from tehillim_pipeline.corpus import Psalm

MIQRABERT_MODEL = "davidmsmiley/MiqraBERT"
ALEPHBERT_MODEL = "imvladikon/sentence-transformers-alephbert"
NEODICTABERT_MODEL = "dicta-il/neodictabert-bilingual-embed"
BEREL_MODEL = "dicta-il/BEREL"
BGE_MULTILINGUAL_GEMMA2_MODEL = "BAAI/bge-multilingual-gemma2"
QWEN3_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
KALM_EMBEDDING_MODEL = "tencent/KaLM-Embedding-Gemma3-12B-2511"
LLAMA_EMBED_NEMOTRON_MODEL = "nvidia/llama-embed-nemotron-8b"
BGE_M3_MODEL = "BAAI/bge-m3"
GTE_MULTILINGUAL_MODEL = "Alibaba-NLP/gte-multilingual-base"

#: This checkpoint's NaN failure mode is intermittent even on CPU;
#: retries, each paired with a dynamic-module eviction, are the actual
#: fix.
_GTE_MULTILINGUAL_MAX_ATTEMPTS = 5

ME5_LARGE_INSTRUCT_MODEL = "intfloat/multilingual-e5-large-instruct"


def _select_half_verses(
    psalm: Psalm, *, vocalized: bool, niqqud_only: bool = False
) -> tuple[str, ...]:
    """Selects a psalm's half-verse texts for the given text state."""
    if niqqud_only:
        return psalm.half_verses_niqqud_only
    return psalm.half_verses if vocalized else psalm.half_verses_unvocalized


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
    """Recomputes this checkpoint's RoPE buffer if it loaded as NaN."""
    import torch

    for module in auto_model.modules():
        freqs_cos = getattr(module, "freqs_cos", None)
        if freqs_cos is None or not torch.isnan(freqs_cos).any():
            continue
        dim = module.config.hidden_size // module.config.num_attention_heads
        cos, sin = _rope_frequencies(dim, module.config.max_length)
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
        if position_ids is None or not torch.is_tensor(position_ids):
            continue
        expected = torch.arange(position_ids.shape[0], device=position_ids.device)
        if torch.equal(position_ids, expected):
            continue
        module.position_ids.copy_(expected)


def _load_kalm_raw_transformer(
    model_name: str,
    *,
    torch_dtype: str | None,
    device: str | None,
    load_tokenizer: Callable[..., Any] | None = None,
    load_model: Callable[..., Any] | None = None,
    cuda_available: Callable[[], bool] | None = None,
    mps_available: Callable[[], bool] | None = None,
) -> tuple[Any, Any]:
    """Loads KaLM via plain transformers: SentenceTransformer crashes on it."""
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

    tokenizer = load_tokenizer(model_name, trust_remote_code=True)
    model_kwargs = {"torch_dtype": torch_dtype} if torch_dtype is not None else {}
    model = load_model(model_name, trust_remote_code=True, **model_kwargs)
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
    """Last-token pooling and L2 normalization for KaLM."""
    import torch

    device = next(model.parameters()).device
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.last_hidden_state
    sequence_lengths = inputs["attention_mask"].sum(dim=1) - 1
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


def compute_half_verse_embeddings(
    psalms: list[Psalm],
    model_name: str,
    *,
    vocalized: bool = True,
    niqqud_only: bool = False,
    device: str | None = None,
    torch_dtype: str | None = None,
    sentence_transformer_factory: Callable[..., Any] | None = None,
    kalm_loader: Callable[..., tuple[Any, Any]] = _load_kalm_raw_transformer,
    last_token_pooler: Callable[[list[str], Any, Any], np.ndarray] = _encode_last_token_pooled,
    release_gpu_memory: Callable[[], None] = _release_gpu_memory,
) -> dict[int, np.ndarray]:
    """`niqqud_only=True` overrides `vocalized` with accent-stripped text."""
    if model_name == KALM_EMBEDDING_MODEL:
        tokenizer, raw_model = kalm_loader(model_name, torch_dtype=torch_dtype, device=device)
        embeddings: dict[int, np.ndarray] = {}
        for psalm in psalms:
            half_verses = _select_half_verses(psalm, vocalized=vocalized, niqqud_only=niqqud_only)
            vectors = last_token_pooler(list(half_verses), tokenizer, raw_model)
            embeddings[psalm.number] = np.asarray(vectors)
        del tokenizer, raw_model
        release_gpu_memory()
        return embeddings

    if sentence_transformer_factory is None:
        from sentence_transformers import SentenceTransformer

        sentence_transformer_factory = SentenceTransformer

    if model_name == GTE_MULTILINGUAL_MODEL and device is None:
        device = "cpu"

    model_kwargs = {"model_kwargs": {"torch_dtype": torch_dtype}} if torch_dtype is not None else {}
    attempts = _GTE_MULTILINGUAL_MAX_ATTEMPTS if model_name == GTE_MULTILINGUAL_MODEL else 1
    for attempt in range(1, attempts + 1):
        if model_name == GTE_MULTILINGUAL_MODEL and attempt > 1:
            _evict_gte_multilingual_dynamic_module()
        model = sentence_transformer_factory(
            model_name, trust_remote_code=True, device=device, **model_kwargs
        )
        if model_name == NEODICTABERT_MODEL:
            _repair_neodictabert_rope_buffers(model[0].auto_model)
        if model_name == GTE_MULTILINGUAL_MODEL:
            _repair_gte_multilingual_position_ids(model[0].auto_model)
        embeddings = {}
        for psalm in psalms:
            half_verses = _select_half_verses(psalm, vocalized=vocalized, niqqud_only=niqqud_only)
            vectors = model.encode(list(half_verses), normalize_embeddings=True)
            embeddings[psalm.number] = np.asarray(vectors)
        del model
        release_gpu_memory()
        if model_name != GTE_MULTILINGUAL_MODEL or all(
            np.all(np.isfinite(v)) for v in embeddings.values()
        ):
            return embeddings
        print(
            f"warning: {model_name} produced non-finite embeddings on attempt "
            f"{attempt}/{attempts}, retrying with a fresh model load",
            file=sys.stderr,
        )
    raise RuntimeError(
        f"{model_name} produced non-finite embeddings on every one of {attempts} attempts"
    )
