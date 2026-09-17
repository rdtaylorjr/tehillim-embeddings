"""Model registry and corpus-data resolution for the Colab-only encoders."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.corpus import BHSA_PATH_ENV, DEFAULT_BHSA_CLONE
from semantic.local_models import (
    BGE_MULTILINGUAL_GEMMA2_MODEL,
    F2LLM_V2_MODEL,
    HARRIER_OSS_MODEL,
    KALM_EMBEDDING_MODEL,
    LLAMA_EMBED_NEMOTRON_MODEL,
    QWEN3_EMBEDDING_MODEL,
)
from semantic.scroll import DEFAULT_DSS_PATH, DEFAULT_SCRIBES_PATH, DSS_PATH_ENV, SCRIBES_PATH_ENV

#: slug -> (model id, torch_dtype); "auto" uses the on-disk dtype rather than upcasting.
LARGE_MODELS: tuple[tuple[str, str, str], ...] = (
    ("bge-multilingual-gemma2", BGE_MULTILINGUAL_GEMMA2_MODEL, "float16"),
    ("qwen3-embedding", QWEN3_EMBEDDING_MODEL, "auto"),
    ("kalm-embedding", KALM_EMBEDDING_MODEL, "bfloat16"),
    ("llama-embed-nemotron", LLAMA_EMBED_NEMOTRON_MODEL, "bfloat16"),
    ("harrier-oss-v1", HARRIER_OSS_MODEL, "auto"),
    ("f2llm-v2", F2LLM_V2_MODEL, "bfloat16"),
)

MODEL_CHOICES: dict[str, str] = {
    "bge": "bge-multilingual-gemma2",
    "qwen3": "qwen3-embedding",
    "kalm": "kalm-embedding",
    "llama-nemotron": "llama-embed-nemotron",
    "harrier": "harrier-oss-v1",
    "f2llm": "f2llm-v2",
}


def models_for_choice(choice: str | None) -> tuple[tuple[str, str, str], ...]:
    """Returns the large models matching `choice`, or all if None."""
    if choice is None:
        return LARGE_MODELS
    slug = MODEL_CHOICES[choice]
    return tuple(model for model in LARGE_MODELS if model[0] == slug)


@dataclass(frozen=True, slots=True)
class Checkout:
    """One Text-Fabric dataset a source reads: its default location and where to clone it from."""

    #: The environment variable the loader reads the location from.
    env_var: str
    default: Path
    repository: str
    #: The Text-Fabric directory inside the clone.
    subpath: str


#: Every dataset the semantic sources read, so a Colab session can fetch what a scope needs.
CHECKOUTS: tuple[Checkout, ...] = (
    Checkout(BHSA_PATH_ENV, DEFAULT_BHSA_CLONE, "https://github.com/ETCBC/bhsa.git", "tf/2021"),
    Checkout(DSS_PATH_ENV, DEFAULT_DSS_PATH, "https://github.com/ETCBC/dss.git", "tf/2.0"),
    Checkout(
        SCRIBES_PATH_ENV,
        DEFAULT_SCRIBES_PATH,
        "https://github.com/rdtaylorjr/tehillim-scribes.git",
        "tf/2.0",
    ),
)


def ensure_checkout(
    checkout: Checkout, *, data_dir: Path, clone: Callable[[str, Path], None]
) -> Path:
    """The dataset's local directory, cloning it into `data_dir` when the default is absent."""
    if checkout.default.exists():
        return checkout.default
    data_dir.mkdir(parents=True, exist_ok=True)
    repository = data_dir / checkout.repository.rsplit("/", 1)[1].removesuffix(".git")
    if not repository.exists():
        clone(checkout.repository, repository)
    return repository / checkout.subpath


def ensure_corpus_data(
    *,
    data_dir: Path,
    clone: Callable[[str, Path], None],
    checkouts: tuple[Checkout, ...] = CHECKOUTS,
) -> dict[str, str]:
    """Every dataset's location, as the environment the sources read: set it before loading."""
    return {
        checkout.env_var: str(ensure_checkout(checkout, data_dir=data_dir, clone=clone))
        for checkout in checkouts
    }


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
