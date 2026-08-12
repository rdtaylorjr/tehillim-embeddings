"""Computes half-verse embeddings from local Hebrew and multilingual
sentence-embedding models.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, MutableMapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch
    from tehillim_pipeline.corpus import Psalm

#: Fine-tuned from AlephBERT for parallel-passage detection (Smiley,
#: arXiv:2606.19638). Its tokenizer strips niqqud and cantillation
#: before the model sees either, inherited from AlephBERT's
#: BertNormalizer.
MIQRABERT_MODEL = "davidmsmiley/MiqraBERT"

#: Unfinetuned AlephBERT, the encoder MiqraBERT was fine-tuned from.
#: Shares MiqraBERT's tokenizer.
ALEPHBERT_MODEL = "imvladikon/sentence-transformers-alephbert"

#: Dicta's sentence-embedding model (Shmidman, Shmidman & Koppel 2025,
#: arXiv:2510.20386). Shares MiqraBERT's tokenizer family.
NEODICTABERT_MODEL = "dicta-il/neodictabert-bilingual-embed"

#: Trained on Rabbinic Hebrew (Shmidman et al. 2022, arXiv:2208.01875).
#: Shares MiqraBERT's tokenizer family.
BEREL_MODEL = "dicta-il/BEREL"

#: MIRACL leader (Zhang et al. 2024, arXiv:2409.15700), fine-tuned from
#: Gemma2-9B. GemmaTokenizer preserves niqqud. Last-token pooling.
BGE_MULTILINGUAL_GEMMA2_MODEL = "BAAI/bge-multilingual-gemma2"

#: MTEB v2 multilingual leader at release (Zhang et al. 2025,
#: arXiv:2506.05176). Qwen2Tokenizer preserves niqqud. Last-token
#: pooling.
QWEN3_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

#: MMTEB multilingual leader (Hu et al. 2025, arXiv:2506.20923).
#: GemmaTokenizer preserves niqqud. Last-token pooling.
KALM_EMBEDDING_MODEL = "tencent/KaLM-Embedding-Gemma3-12B-2511"

#: Multilingual MTEB leader as of October 2025 (Llama-Embed-Nemotron
#: Team, arXiv:2511.07025), fine-tuned from Llama-3.1-8B with
#: bidirectional attention. Mean pooling. Licensed for non-commercial,
#: research use only.
LLAMA_EMBED_NEMOTRON_MODEL = "nvidia/llama-embed-nemotron-8b"

#: ~568M params, XLM-RoBERTa-large based (Chen et al. 2024,
#: arXiv:2402.03216). XLMRobertaTokenizer preserves niqqud. CLS-token
#: pooling.
BGE_M3_MODEL = "BAAI/bge-m3"

#: ~305M params, RoPE-based bidirectional encoder (Zhang et al. 2024,
#: arXiv:2407.19669). XLMRobertaTokenizer preserves niqqud. CLS-token
#: pooling. Requires trust_remote_code.
GTE_MULTILINGUAL_MODEL = "Alibaba-NLP/gte-multilingual-base"

#: This checkpoint's NaN failure mode is intermittent even on CPU;
#: retries, each paired with a dynamic-module eviction, are the actual
#: fix.
_GTE_MULTILINGUAL_MAX_ATTEMPTS = 5

#: ~560M params, XLM-RoBERTa-large based (Wang et al. 2024,
#: arXiv:2402.05672). XLMRobertaTokenizer preserves niqqud. Mean-token
#: pooling.
ME5_LARGE_INSTRUCT_MODEL = "intfloat/multilingual-e5-large-instruct"


def _select_half_verses(
    psalm: Psalm, *, vocalized: bool, niqqud_only: bool = False
) -> tuple[str, ...]:
    if niqqud_only:
        return psalm.half_verses_niqqud_only
    return psalm.half_verses if vocalized else psalm.half_verses_unvocalized


def _rope_frequencies(
    dim: int, length: int, theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reimplements NeoDictaBERT's own `precompute_freqs`
    (`modeling_neobert.py`, dicta-il/neodictabert-bilingual-embed): a
    pure function of dim/length/theta, no learned weights.
    """
    import torch

    half = dim // 2
    idx = torch.arange(0, half, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** ((2.0 * idx) / dim))
    positions = torch.arange(length, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    return angles.cos(), angles.sin()


def _repair_neodictabert_rope_buffers(auto_model: Any) -> None:
    """NeoDictaBERT computes its RoPE frequency tables once at init and
    stores them as a non-persistent buffer, reused by every later
    encode() call. Measured directly: roughly half of fresh loads leave
    that buffer NaN, on CPU and MPS alike, poisoning every subsequent
    embedding until the process exits. Detects a NaN buffer and
    recomputes it with `_rope_frequencies`, which is mathematically
    identical to what a clean init produces.
    """
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
    """Purges GTE-multilingual-base's trust_remote_code dynamic module
    from sys.modules, forcing a real reimport on the next
    SentenceTransformer(..., trust_remote_code=True) call. Once this
    checkpoint's NaN failure hits once in a process, a fresh
    SentenceTransformer object alone doesn't recover: the dynamic
    module stays cached in sys.modules and every in-process retry
    shares whatever state got corrupted on the first failure.
    """
    if modules is None:
        modules = sys.modules
    stale = [name for name in modules if "transformers_modules.Alibaba_hyphen_NLP" in name]
    for name in stale:
        del modules[name]


def _repair_gte_multilingual_position_ids(auto_model: Any) -> None:
    """GTE-multilingual-base registers a non-persistent `position_ids`
    buffer at init, reused by every later encode() call. Measured
    directly: a fresh load's buffer sometimes comes back as arbitrary
    garbage integers rather than torch.arange(n), crashing every
    encode() with an out-of-bounds index. Detects a buffer that doesn't
    match a clean torch.arange and overwrites it in place.
    """
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
    """Loads KaLM-Embedding-Gemma3-12B via plain transformers instead of
    SentenceTransformer, whose Transformer module calls
    AutoProcessor.from_pretrained unconditionally; this checkpoint ships
    no preprocessor_config.json, so that call always fails.
    """
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
    """Last-token pooling and L2 normalization, applied manually since
    KaLM bypasses SentenceTransformer.
    """
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
    """Encode every psalm's half-verses with `model_name`.

    `model_name == KALM_EMBEDDING_MODEL` routes through `kalm_loader`/
    `last_token_pooler` instead of SentenceTransformer (see
    `_load_kalm_raw_transformer`).

    `vocalized=False` encodes the consonantal-only text instead of the
    default fully vocalized text. `niqqud_only=True` overrides
    `vocalized` entirely and encodes the niqqud-preserved,
    accent-stripped text instead.

    `device` defaults to None (auto-detect). `GTE_MULTILINGUAL_MODEL` is
    forced to CPU when `device` is None: it produces all-NaN output on
    MPS under memory pressure. CPU alone isn't fully reliable either, so
    it additionally retries up to `_GTE_MULTILINGUAL_MAX_ATTEMPTS` times
    with a fresh model load if any output is non-finite.

    `torch_dtype` defaults to None (omits `model_kwargs` entirely),
    passed through for the heavy encoders that need explicit fp16/bf16
    loading to fit in memory.

    `sentence_transformer_factory` defaults to the real
    `sentence_transformers.SentenceTransformer`.

    Returns one L2-normalized embedding matrix per psalm number, shape
    (n_half_verses, embedding_dim).
    """
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
