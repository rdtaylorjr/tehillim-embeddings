"""Ordering and batching guarantees for the OpenRouter-backed fetchers, with no network calls."""

from __future__ import annotations

import numpy as np
import pytest

from semantic.api_models import (
    GEMINI_BATCH_SIZE,
    GEMINI_DIMENSIONS,
    fetch_gemini_embeddings,
)


def _vector(lead: float, dim: int) -> list[float]:
    return [lead] + [0.0] * (dim - 1)


class _Embedding:
    def __init__(self, embedding: list[float], index: int) -> None:
        self.embedding = embedding
        self.index = index


class _Response:
    def __init__(self, data: list[_Embedding]) -> None:
        self.data = data


class _ReversingEmbeddings:
    """Returns each batch's embeddings in reverse order, as the API is permitted to."""

    def __init__(self, dim: int, batches: list[list[str]]) -> None:
        self._dim = dim
        self._batches = batches

    def create(self, **kwargs):
        texts = kwargs["input"]
        self._batches.append(list(texts))
        data = [
            _Embedding(_vector(float(len(text)), self._dim), index)
            for index, text in enumerate(texts)
        ]
        return _Response(list(reversed(data)))


class _ReversingClient:
    def __init__(self, *, api_key: str, base_url: str) -> None:
        self.batches: list[list[str]] = []
        self.embeddings = _ReversingEmbeddings(GEMINI_DIMENSIONS, self.batches)


class _DuplicateIndexEmbeddings:
    def __init__(self, dim: int) -> None:
        self._dim = dim

    def create(self, **kwargs):
        texts = kwargs["input"]
        return _Response([_Embedding(_vector(float(len(t)), self._dim), 0) for t in texts])


class _DuplicateIndexClient:
    def __init__(self, *, api_key: str, base_url: str) -> None:
        self.embeddings = _DuplicateIndexEmbeddings(GEMINI_DIMENSIONS)


class _ShortBatchEmbeddings:
    def __init__(self, dim: int) -> None:
        self._dim = dim

    def create(self, **kwargs):
        texts = kwargs["input"]
        data = [_Embedding(_vector(float(len(t)), self._dim), i) for i, t in enumerate(texts[:-1])]
        return _Response(data)


class _ShortBatchClient:
    def __init__(self, *, api_key: str, base_url: str) -> None:
        self.embeddings = _ShortBatchEmbeddings(GEMINI_DIMENSIONS)


class TestResponseOrdering:
    def test_reorders_a_batch_returned_out_of_order_back_to_input_order(self):
        texts = ["a", "bb", "ccc", "dddd"]

        vectors = fetch_gemini_embeddings(
            texts, api_key="k", client_factory=_ReversingClient, max_workers=1
        )

        assert vectors[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_keeps_input_order_across_several_batches(self):
        texts = [f"{'x' * (i % 7 + 1)}" for i in range(GEMINI_BATCH_SIZE * 3 + 5)]

        vectors = fetch_gemini_embeddings(
            texts, api_key="k", client_factory=_ReversingClient, max_workers=1
        )

        assert vectors[:, 0].tolist() == [float(len(text)) for text in texts]

    def test_returns_one_vector_per_input_text(self):
        texts = ["a"] * (GEMINI_BATCH_SIZE + 1)

        vectors = fetch_gemini_embeddings(
            texts, api_key="k", client_factory=_ReversingClient, max_workers=1
        )

        assert vectors.shape == (GEMINI_BATCH_SIZE + 1, GEMINI_DIMENSIONS)

    def test_parallel_batches_produce_the_same_order_as_serial_batches(self):
        texts = [f"{'x' * (i % 9 + 1)}" for i in range(GEMINI_BATCH_SIZE * 4)]

        serial = fetch_gemini_embeddings(
            texts, api_key="k", client_factory=_ReversingClient, max_workers=1
        )
        parallel = fetch_gemini_embeddings(
            texts, api_key="k", client_factory=_ReversingClient, max_workers=4
        )

        assert np.array_equal(serial, parallel)

    def test_rejects_a_batch_whose_indices_are_not_a_permutation_of_its_inputs(self):
        with pytest.raises(RuntimeError, match="indices"):
            fetch_gemini_embeddings(
                ["a", "bb"], api_key="k", client_factory=_DuplicateIndexClient, max_workers=1
            )

    def test_rejects_a_batch_that_returns_fewer_embeddings_than_texts(self):
        with pytest.raises(RuntimeError, match="2 embeddings for a batch of 3"):
            fetch_gemini_embeddings(
                ["a", "bb", "ccc"], api_key="k", client_factory=_ShortBatchClient, max_workers=1
            )

    def test_an_empty_input_never_calls_the_api(self):
        calls: list[list[str]] = []

        class _Client:
            def __init__(self, *, api_key: str, base_url: str) -> None:
                self.embeddings = _ReversingEmbeddings(GEMINI_DIMENSIONS, calls)

        vectors = fetch_gemini_embeddings([], api_key="k", client_factory=_Client)

        assert calls == []
        assert vectors.shape == (0, GEMINI_DIMENSIONS)
