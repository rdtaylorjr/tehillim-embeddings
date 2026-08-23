"""Fetches colon embeddings from Gemini, OpenAI, Cohere, and Voyage."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

#: Real pricing entry, 8192 context length, confirmed against
#: OpenRouter's /embeddings/models endpoint.
GEMINI_MODEL = "google/gemini-embedding-2"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: Native dimensionality. `fetch_gemini_embeddings` never passes
#: `dimensions`, so output should never be truncated below this.
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


def fetch_gemini_embeddings(
    texts: list[str], *, api_key: str, client_factory: Callable[..., Any] | None = None
) -> np.ndarray:
    """Fetches embeddings for `texts` from Gemini Embedding 2."""
    if client_factory is None:
        client_factory = _real_openai_client()

    client = client_factory(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), GEMINI_BATCH_SIZE):
        batch = texts[start : start + GEMINI_BATCH_SIZE]
        response = client.embeddings.create(
            model=GEMINI_MODEL, input=batch, encoding_format="float"
        )
        if not response.data:
            raise RuntimeError(
                f"OpenRouter embeddings.create (model={GEMINI_MODEL}) returned no "
                f"embeddings for a batch of {len(batch)} texts"
            )
        for embedding in response.data:
            if len(embedding.embedding) != GEMINI_DIMENSIONS:
                raise RuntimeError(
                    f"OpenRouter embeddings.create (model={GEMINI_MODEL}) returned a "
                    f"{len(embedding.embedding)}-dimensional vector, expected "
                    f"{GEMINI_DIMENSIONS}."
                )
            vectors.append(embedding.embedding)
    return np.asarray(vectors, dtype=np.float64)


def fetch_cohere_embeddings(
    texts: list[str], *, api_key: str, client_factory: Callable[..., Any] | None = None
) -> np.ndarray:
    """Fetches embeddings for `texts` from Cohere Embed v4."""
    if client_factory is None:
        client_factory = _real_cohere_client()

    client = client_factory(api_key=api_key)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), COHERE_BATCH_SIZE):
        batch = texts[start : start + COHERE_BATCH_SIZE]
        response = client.embed(
            model=COHERE_MODEL,
            input_type=COHERE_INPUT_TYPE,
            texts=batch,
            embedding_types=["float"],
        )
        batch_vectors = response.embeddings.float_
        if not batch_vectors:
            raise RuntimeError(
                f"Cohere embed (model={COHERE_MODEL}) returned no embeddings for a "
                f"batch of {len(batch)} texts"
            )
        for embedding in batch_vectors:
            if len(embedding) != COHERE_DIMENSIONS:
                raise RuntimeError(
                    f"Cohere embed (model={COHERE_MODEL}) returned a "
                    f"{len(embedding)}-dimensional vector, expected {COHERE_DIMENSIONS}."
                )
            vectors.append(list(embedding))
    return np.asarray(vectors, dtype=np.float64)


def fetch_openai_embeddings(
    texts: list[str], *, api_key: str, client_factory: Callable[..., Any] | None = None
) -> np.ndarray:
    """Fetches embeddings for `texts` from OpenAI text-embedding-3-large."""
    if client_factory is None:
        client_factory = _real_openai_client()

    client = client_factory(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), OPENAI_BATCH_SIZE):
        batch = texts[start : start + OPENAI_BATCH_SIZE]
        response = client.embeddings.create(
            model=OPENAI_MODEL, input=batch, encoding_format="float"
        )
        if not response.data:
            raise RuntimeError(
                f"OpenRouter embeddings.create (model={OPENAI_MODEL}) returned no "
                f"embeddings for a batch of {len(batch)} texts"
            )
        for embedding in response.data:
            if len(embedding.embedding) != OPENAI_DIMENSIONS:
                raise RuntimeError(
                    f"OpenRouter embeddings.create (model={OPENAI_MODEL}) returned a "
                    f"{len(embedding.embedding)}-dimensional vector, expected "
                    f"{OPENAI_DIMENSIONS}."
                )
            vectors.append(embedding.embedding)
    return np.asarray(vectors, dtype=np.float64)


def fetch_voyage_embeddings(
    texts: list[str], *, api_key: str, client_factory: Callable[..., Any] | None = None
) -> np.ndarray:
    """Fetches embeddings for `texts` from Voyage 4 via OpenRouter."""
    if client_factory is None:
        client_factory = _real_openai_client()

    client = client_factory(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), VOYAGE_BATCH_SIZE):
        batch = texts[start : start + VOYAGE_BATCH_SIZE]
        response = client.embeddings.create(
            model=VOYAGE_MODEL, input=batch, encoding_format="float"
        )
        if not response.data:
            raise RuntimeError(
                f"OpenRouter embeddings.create (model={VOYAGE_MODEL}) returned no "
                f"embeddings for a batch of {len(batch)} texts"
            )
        for embedding in response.data:
            if len(embedding.embedding) != VOYAGE_DIMENSIONS:
                raise RuntimeError(
                    f"OpenRouter embeddings.create (model={VOYAGE_MODEL}) returned a "
                    f"{len(embedding.embedding)}-dimensional vector, expected "
                    f"{VOYAGE_DIMENSIONS}."
                )
            vectors.append(embedding.embedding)
    return np.asarray(vectors, dtype=np.float64)


#: OpenRouter issues one key per account, not per underlying provider.
API_KEY_ENV_VARS = {
    "gemini": "TEHILLIM_OPENROUTER_API_KEY",
    "cohere": "TEHILLIM_COHERE_API_KEY",
    "openai": "TEHILLIM_OPENROUTER_API_KEY",
    "voyage": "TEHILLIM_OPENROUTER_API_KEY",
}
