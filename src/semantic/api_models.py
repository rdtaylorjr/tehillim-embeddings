"""Fetches half-verse embeddings from Gemini, OpenAI, Cohere, and Voyage."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: Concurrent in-flight batches, kept modest to stay inside provider rate limits.
DEFAULT_MAX_WORKERS = 4

#: Real pricing entry, 8192 context length, per OpenRouter's /embeddings/models endpoint.
GEMINI_MODEL = "google/gemini-embedding-2"

#: Native dimensionality; `fetch_gemini_embeddings` never passes `dimensions`.
GEMINI_DIMENSIONS = 3072

GEMINI_BATCH_SIZE = 100

COHERE_MODEL = "embed-v4.0"

#: Cohere's default when `output_dimension` is never passed.
COHERE_DIMENSIONS = 1536

#: Cohere's documented hard per-call limit.
COHERE_BATCH_SIZE = 96

#: Required by Cohere with no default.
COHERE_INPUT_TYPE = "search_document"

OPENAI_MODEL = "openai/text-embedding-3-large"

OPENAI_DIMENSIONS = 3072

OPENAI_BATCH_SIZE = 100

VOYAGE_MODEL = "voyageai/voyage-4"

#: Voyage's default when `output_dimension` is never passed.
VOYAGE_DIMENSIONS = 1024

VOYAGE_BATCH_SIZE = 100


def _real_openai_client() -> Callable[..., Any]:
    """Returns the real `openai.OpenAI` client class."""
    import openai

    return openai.OpenAI


def _real_cohere_client() -> Callable[..., Any]:
    """Returns the real `cohere.ClientV2` client class."""
    import cohere

    return cohere.ClientV2


@dataclass(frozen=True, slots=True)
class _OpenRouterModel:
    """One OpenRouter-hosted embedding model: its id, native dimensionality, and batch size."""

    model: str
    dimensions: int
    batch_size: int


def _batches(texts: list[str], size: int) -> list[list[str]]:
    """Splits `texts` into consecutive batches of at most `size`, preserving order."""
    return [texts[start : start + size] for start in range(0, len(texts), size)]


def _gather(
    batches: list[list[str]],
    fetch: Callable[[list[str]], list[list[float]]],
    max_workers: int,
) -> np.ndarray:
    """Fetches every batch, concurrently when it helps, concatenated in batch order."""
    workers = max(1, min(max_workers, len(batches)))
    if workers == 1:
        per_batch = [fetch(batch) for batch in batches]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            per_batch = list(pool.map(fetch, batches))
    return np.asarray([vector for batch in per_batch for vector in batch], dtype=np.float64)


def _ordered_batch_vectors(
    response: Any, batch: list[str], spec: _OpenRouterModel
) -> list[list[float]]:
    """Restores input order from the response's `index` field and validates dimensionality."""
    data = list(response.data or ())
    if not data:
        raise RuntimeError(
            f"OpenRouter embeddings.create (model={spec.model}) returned no "
            f"embeddings for a batch of {len(batch)} texts"
        )
    if len(data) != len(batch):
        raise RuntimeError(
            f"OpenRouter embeddings.create (model={spec.model}) returned {len(data)} "
            f"embeddings for a batch of {len(batch)} texts"
        )
    if sorted(item.index for item in data) != list(range(len(batch))):
        raise RuntimeError(
            f"OpenRouter embeddings.create (model={spec.model}) returned indices that are "
            f"not a permutation of 0..{len(batch) - 1}, so input order cannot be restored"
        )
    for item in data:
        if len(item.embedding) != spec.dimensions:
            raise RuntimeError(
                f"OpenRouter embeddings.create (model={spec.model}) returned a "
                f"{len(item.embedding)}-dimensional vector, expected {spec.dimensions}."
            )
    return [item.embedding for item in sorted(data, key=lambda item: item.index)]


def _fetch_openrouter_embeddings(
    texts: list[str],
    spec: _OpenRouterModel,
    *,
    api_key: str,
    client_factory: Callable[..., Any] | None,
    max_workers: int,
) -> np.ndarray:
    """Fetches embeddings batch-parallel, reassembling them in the caller's input order."""
    if not texts:
        return np.zeros((0, spec.dimensions), dtype=np.float64)
    if client_factory is None:
        client_factory = _real_openai_client()

    client = client_factory(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    batches = _batches(texts, spec.batch_size)

    def fetch(batch: list[str]) -> list[list[float]]:
        """Fetches and order-restores one batch."""
        response = client.embeddings.create(model=spec.model, input=batch, encoding_format="float")
        return _ordered_batch_vectors(response, batch, spec)

    return _gather(batches, fetch, max_workers)


_GEMINI = _OpenRouterModel(GEMINI_MODEL, GEMINI_DIMENSIONS, GEMINI_BATCH_SIZE)
_OPENAI = _OpenRouterModel(OPENAI_MODEL, OPENAI_DIMENSIONS, OPENAI_BATCH_SIZE)
_VOYAGE = _OpenRouterModel(VOYAGE_MODEL, VOYAGE_DIMENSIONS, VOYAGE_BATCH_SIZE)


def fetch_gemini_embeddings(
    texts: list[str],
    *,
    api_key: str,
    client_factory: Callable[..., Any] | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> np.ndarray:
    """Fetches embeddings for `texts` from Gemini Embedding 2."""
    return _fetch_openrouter_embeddings(
        texts, _GEMINI, api_key=api_key, client_factory=client_factory, max_workers=max_workers
    )


def fetch_openai_embeddings(
    texts: list[str],
    *,
    api_key: str,
    client_factory: Callable[..., Any] | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> np.ndarray:
    """Fetches embeddings for `texts` from OpenAI text-embedding-3-large."""
    return _fetch_openrouter_embeddings(
        texts, _OPENAI, api_key=api_key, client_factory=client_factory, max_workers=max_workers
    )


def fetch_voyage_embeddings(
    texts: list[str],
    *,
    api_key: str,
    client_factory: Callable[..., Any] | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> np.ndarray:
    """Fetches embeddings for `texts` from Voyage 4 via OpenRouter."""
    return _fetch_openrouter_embeddings(
        texts, _VOYAGE, api_key=api_key, client_factory=client_factory, max_workers=max_workers
    )


def fetch_cohere_embeddings(
    texts: list[str],
    *,
    api_key: str,
    client_factory: Callable[..., Any] | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> np.ndarray:
    """Fetches embeddings for `texts` from Cohere Embed v4."""
    if not texts:
        return np.zeros((0, COHERE_DIMENSIONS), dtype=np.float64)
    if client_factory is None:
        client_factory = _real_cohere_client()

    client = client_factory(api_key=api_key)
    batches = _batches(texts, COHERE_BATCH_SIZE)

    def fetch(batch: list[str]) -> list[list[float]]:
        """Fetches and validates one batch."""
        response = client.embed(
            model=COHERE_MODEL,
            input_type=COHERE_INPUT_TYPE,
            texts=batch,
            embedding_types=["float"],
        )
        #: Cohere returns embeddings positionally, with no index field to sort by.
        batch_vectors = response.embeddings.float_
        if not batch_vectors:
            raise RuntimeError(
                f"Cohere embed (model={COHERE_MODEL}) returned no embeddings for a "
                f"batch of {len(batch)} texts"
            )
        if len(batch_vectors) != len(batch):
            raise RuntimeError(
                f"Cohere embed (model={COHERE_MODEL}) returned {len(batch_vectors)} "
                f"embeddings for a batch of {len(batch)} texts"
            )
        for embedding in batch_vectors:
            if len(embedding) != COHERE_DIMENSIONS:
                raise RuntimeError(
                    f"Cohere embed (model={COHERE_MODEL}) returned a "
                    f"{len(embedding)}-dimensional vector, expected {COHERE_DIMENSIONS}."
                )
        return [list(embedding) for embedding in batch_vectors]

    return _gather(batches, fetch, max_workers)


#: OpenRouter issues one key per account, not per underlying provider.
API_KEY_ENV_VARS = {
    "gemini": "TEHILLIM_OPENROUTER_API_KEY",
    "cohere": "TEHILLIM_COHERE_API_KEY",
    "openai": "TEHILLIM_OPENROUTER_API_KEY",
    "voyage": "TEHILLIM_OPENROUTER_API_KEY",
}
