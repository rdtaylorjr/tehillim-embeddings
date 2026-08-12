"""Model registry and corpus-data resolution for the four Colab-only encoders."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from semantic.local_models import (
    BGE_MULTILINGUAL_GEMMA2_MODEL,
    KALM_EMBEDDING_MODEL,
    LLAMA_EMBED_NEMOTRON_MODEL,
    QWEN3_EMBEDDING_MODEL,
)

#: slug -> (technical model id, torch_dtype). torch_dtype="auto" uses
#: the checkpoint's on-disk dtype rather than upcasting to fp32.
LARGE_MODELS: tuple[tuple[str, str, str], ...] = (
    ("bge-multilingual-gemma2", BGE_MULTILINGUAL_GEMMA2_MODEL, "float16"),
    ("qwen3-embedding", QWEN3_EMBEDDING_MODEL, "auto"),
    ("kalm-embedding", KALM_EMBEDDING_MODEL, "bfloat16"),
    ("llama-embed-nemotron", LLAMA_EMBED_NEMOTRON_MODEL, "bfloat16"),
)

MODEL_CHOICES: dict[str, str] = {
    "bge": "bge-multilingual-gemma2",
    "qwen3": "qwen3-embedding",
    "kalm": "kalm-embedding",
    "llama-nemotron": "llama-embed-nemotron",
}


def models_for_choice(choice: str | None) -> tuple[tuple[str, str, str], ...]:
    """Returns the large models matching `choice`, or all if None."""
    if choice is None:
        return LARGE_MODELS
    slug = MODEL_CHOICES[choice]
    return tuple(model for model in LARGE_MODELS if model[0] == slug)


def ensure_corpus_data(
    *,
    bhsa_default: Path,
    data_dir: Path,
    clone: Callable[[str, Path], None],
) -> Path:
    """Returns a local BHSA path, cloning into `data_dir` if missing."""
    if bhsa_default.exists():
        return bhsa_default

    data_dir.mkdir(parents=True, exist_ok=True)
    bhsa_repo = data_dir / "bhsa"
    if not bhsa_repo.exists():
        clone("https://github.com/ETCBC/bhsa.git", bhsa_repo)
    return bhsa_repo / "tf" / "2021"


def gpu_memory_summary(torch_module: Any | None = None) -> str | None:
    """Returns a formatted CUDA memory summary, or None if unavailable."""
    resolved: Any = torch_module
    if resolved is None:
        try:
            import torch

            resolved = torch
        except ImportError:
            return None
    if not resolved.cuda.is_available():
        return None
    allocated = resolved.cuda.memory_allocated() / 1e9
    reserved = resolved.cuda.memory_reserved() / 1e9
    total = resolved.cuda.get_device_properties(0).total_memory / 1e9
    return f"allocated={allocated:.2f}GB reserved={reserved:.2f}GB total={total:.2f}GB"
