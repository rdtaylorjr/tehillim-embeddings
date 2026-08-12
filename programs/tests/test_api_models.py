"""Unit tests for api_models.py. Fakes `client_factory`, makes no real network calls."""

from __future__ import annotations

import numpy as np
import pytest

from semantic.api_models import (
    fetch_cohere_embeddings,
    fetch_gemini_embeddings,
    fetch_openai_embeddings,
    fetch_voyage_embeddings,
)


def _full_dim_vector(lead_value: float, dim: int) -> list[float]:
    return [lead_value] + [0.0] * (dim - 1)


class _FakeEmbedding:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, data: list[_FakeEmbedding] | None) -> None:
        self.data = data


class _FakeEmbeddingsResource:
    def __init__(self, calls: list[dict[str, object]], dim: int) -> None:
        self._calls = calls
        self._dim = dim

    def create(self, **kwargs: object) -> _FakeEmbeddingResponse:
        self._calls.append(kwargs)
        texts = kwargs["input"]
        assert isinstance(texts, list)
        return _FakeEmbeddingResponse(
            data=[_FakeEmbedding(_full_dim_vector(float(len(text)), self._dim)) for text in texts]
        )


class _FakeOpenAiCompatibleClient:
    def __init__(self, *, api_key: str, base_url: str, dim: int = 3072) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.calls: list[dict[str, object]] = []
        self.embeddings = _FakeEmbeddingsResource(self.calls, dim)


class _FakeCohereEmbeddings:
    def __init__(self, float_: list[list[float]] | None) -> None:
        # The real attribute is `float_` (trailing underscore, aliased
        # from the JSON key "float"), confirmed against cohere-python's
        # EmbedByTypeResponseEmbeddings type.
        self.float_ = float_


class _FakeCohereEmbedResponse:
    def __init__(self, float_: list[list[float]] | None) -> None:
        self.embeddings = _FakeCohereEmbeddings(float_)


class _FakeCohereClient:
    def __init__(self, *, api_key: str, response_dim: int = 1536) -> None:
        self.api_key = api_key
        self.calls: list[dict[str, object]] = []
        self._response_dim = response_dim

    def embed(self, **kwargs: object) -> _FakeCohereEmbedResponse:
        self.calls.append(kwargs)
        texts = kwargs["texts"]
        assert isinstance(texts, list)
        return _FakeCohereEmbedResponse(
            float_=[_full_dim_vector(float(len(text)), self._response_dim) for text in texts]
        )


class TestFetchGeminiEmbeddings:
    def test_calls_the_client_with_the_real_model_id_and_never_passes_dimensions(self):
        created: list[_FakeOpenAiCompatibleClient] = []

        def _factory(*, api_key: str, base_url: str) -> _FakeOpenAiCompatibleClient:
            client = _FakeOpenAiCompatibleClient(api_key=api_key, base_url=base_url)
            created.append(client)
            return client

        fetch_gemini_embeddings(["שלום", "עולם"], api_key="secret-key", client_factory=_factory)

        assert len(created) == 1
        assert created[0].api_key == "secret-key"
        assert created[0].base_url == "https://openrouter.ai/api/v1"
        [call] = created[0].calls
        assert call["model"] == "google/gemini-embedding-2"
        assert call["input"] == ["שלום", "עולם"]
        assert "dimensions" not in call

    def test_batches_texts_at_the_documented_batch_size(self):
        created: list[_FakeOpenAiCompatibleClient] = []

        def _factory(*, api_key: str, base_url: str) -> _FakeOpenAiCompatibleClient:
            client = _FakeOpenAiCompatibleClient(api_key=api_key, base_url=base_url)
            created.append(client)
            return client

        texts = [f"text-{i}" for i in range(250)]
        fetch_gemini_embeddings(texts, api_key="k", client_factory=_factory)

        batch_sizes = [len(call["input"]) for call in created[0].calls]
        assert batch_sizes == [100, 100, 50]

    def test_returns_one_vector_per_text_in_order(self):
        def _factory(*, api_key: str, base_url: str) -> _FakeOpenAiCompatibleClient:
            return _FakeOpenAiCompatibleClient(api_key=api_key, base_url=base_url)

        result = fetch_gemini_embeddings(["a", "bb", "ccc"], api_key="k", client_factory=_factory)

        assert result.shape == (3, 3072)
        assert np.allclose(result[:, 0], [1.0, 2.0, 3.0])

    def test_raises_a_clear_error_if_the_api_returns_no_data(self):
        class _EmptyEmbeddingsResource:
            def create(self, **kwargs: object) -> _FakeEmbeddingResponse:
                return _FakeEmbeddingResponse(data=None)

        class _EmptyClient:
            def __init__(self, *, api_key: str, base_url: str) -> None:
                self.embeddings = _EmptyEmbeddingsResource()

        with pytest.raises(RuntimeError, match="no embeddings"):
            fetch_gemini_embeddings(["a"], api_key="k", client_factory=_EmptyClient)

    def test_raises_a_clear_error_if_a_returned_embedding_is_not_3072_dimensional(self):
        class _WrongDimEmbeddingsResource:
            def create(self, **kwargs: object) -> _FakeEmbeddingResponse:
                return _FakeEmbeddingResponse(data=[_FakeEmbedding([0.0] * 768)])

        class _WrongDimClient:
            def __init__(self, *, api_key: str, base_url: str) -> None:
                self.embeddings = _WrongDimEmbeddingsResource()

        with pytest.raises(RuntimeError, match="3072"):
            fetch_gemini_embeddings(["a"], api_key="k", client_factory=_WrongDimClient)


class TestFetchCohereEmbeddings:
    def test_calls_the_client_with_the_real_model_id_and_input_type(self):
        created: list[_FakeCohereClient] = []

        def _factory(*, api_key: str) -> _FakeCohereClient:
            client = _FakeCohereClient(api_key=api_key)
            created.append(client)
            return client

        fetch_cohere_embeddings(["שלום", "עולם"], api_key="secret-key", client_factory=_factory)

        assert len(created) == 1
        assert created[0].api_key == "secret-key"
        [call] = created[0].calls
        assert call["model"] == "embed-v4.0"
        assert call["input_type"] == "search_document"
        assert call["texts"] == ["שלום", "עולם"]
        assert call["embedding_types"] == ["float"]

    def test_batches_texts_at_cohere_s_documented_96_text_limit(self):
        created: list[_FakeCohereClient] = []

        def _factory(*, api_key: str) -> _FakeCohereClient:
            client = _FakeCohereClient(api_key=api_key)
            created.append(client)
            return client

        texts = [f"text-{i}" for i in range(250)]
        fetch_cohere_embeddings(texts, api_key="k", client_factory=_factory)

        batch_sizes = [len(call["texts"]) for call in created[0].calls]
        assert batch_sizes == [96, 96, 58]

    def test_returns_one_vector_per_text_in_order(self):
        result = fetch_cohere_embeddings(
            ["a", "bb", "ccc"], api_key="k", client_factory=_FakeCohereClient
        )

        assert result.shape == (3, 1536)
        assert np.allclose(result[:, 0], [1.0, 2.0, 3.0])

    def test_raises_a_clear_error_if_the_api_returns_no_embeddings(self):
        class _EmptyCohereClient:
            def __init__(self, *, api_key: str) -> None:
                pass

            def embed(self, **kwargs: object) -> _FakeCohereEmbedResponse:
                return _FakeCohereEmbedResponse(float_=None)

        with pytest.raises(RuntimeError, match="no embeddings"):
            fetch_cohere_embeddings(["a"], api_key="k", client_factory=_EmptyCohereClient)

    def test_raises_a_clear_error_if_a_returned_embedding_is_not_1536_dimensional(self):
        class _WrongDimCohereClient:
            def __init__(self, *, api_key: str) -> None:
                pass

            def embed(self, **kwargs: object) -> _FakeCohereEmbedResponse:
                return _FakeCohereEmbedResponse(float_=[[0.0] * 768])

        with pytest.raises(RuntimeError, match="1536"):
            fetch_cohere_embeddings(["a"], api_key="k", client_factory=_WrongDimCohereClient)


class TestFetchOpenaiEmbeddings:
    def test_calls_the_client_with_the_real_model_id_and_never_passes_dimensions(self):
        created: list[_FakeOpenAiCompatibleClient] = []

        def _factory(*, api_key: str, base_url: str) -> _FakeOpenAiCompatibleClient:
            client = _FakeOpenAiCompatibleClient(api_key=api_key, base_url=base_url)
            created.append(client)
            return client

        fetch_openai_embeddings(["שלום", "עולם"], api_key="secret-key", client_factory=_factory)

        assert len(created) == 1
        [call] = created[0].calls
        assert call["model"] == "openai/text-embedding-3-large"
        assert "dimensions" not in call

    def test_batches_texts_at_the_documented_batch_size(self):
        created: list[_FakeOpenAiCompatibleClient] = []

        def _factory(*, api_key: str, base_url: str) -> _FakeOpenAiCompatibleClient:
            client = _FakeOpenAiCompatibleClient(api_key=api_key, base_url=base_url)
            created.append(client)
            return client

        texts = [f"text-{i}" for i in range(250)]
        fetch_openai_embeddings(texts, api_key="k", client_factory=_factory)

        batch_sizes = [len(call["input"]) for call in created[0].calls]
        assert batch_sizes == [100, 100, 50]

    def test_raises_a_clear_error_if_a_returned_embedding_is_not_3072_dimensional(self):
        class _WrongDimEmbeddingsResource:
            def create(self, **kwargs: object) -> _FakeEmbeddingResponse:
                return _FakeEmbeddingResponse(data=[_FakeEmbedding([0.0] * 768)])

        class _WrongDimClient:
            def __init__(self, *, api_key: str, base_url: str) -> None:
                self.embeddings = _WrongDimEmbeddingsResource()

        with pytest.raises(RuntimeError, match="3072"):
            fetch_openai_embeddings(["a"], api_key="k", client_factory=_WrongDimClient)


class TestFetchVoyageEmbeddings:
    def test_calls_the_client_with_the_real_model_id_and_never_passes_dimensions(self):
        created: list[_FakeOpenAiCompatibleClient] = []

        def _factory(*, api_key: str, base_url: str) -> _FakeOpenAiCompatibleClient:
            client = _FakeOpenAiCompatibleClient(api_key=api_key, base_url=base_url, dim=1024)
            created.append(client)
            return client

        fetch_voyage_embeddings(["שלום", "עולם"], api_key="secret-key", client_factory=_factory)

        assert len(created) == 1
        [call] = created[0].calls
        assert call["model"] == "voyageai/voyage-4"
        assert "dimensions" not in call

    def test_batches_texts_at_the_documented_batch_size(self):
        created: list[_FakeOpenAiCompatibleClient] = []

        def _factory(*, api_key: str, base_url: str) -> _FakeOpenAiCompatibleClient:
            client = _FakeOpenAiCompatibleClient(api_key=api_key, base_url=base_url, dim=1024)
            created.append(client)
            return client

        texts = [f"text-{i}" for i in range(250)]
        fetch_voyage_embeddings(texts, api_key="k", client_factory=_factory)

        batch_sizes = [len(call["input"]) for call in created[0].calls]
        assert batch_sizes == [100, 100, 50]

    def test_returns_one_vector_per_text_in_order(self):
        def _factory(*, api_key: str, base_url: str) -> _FakeOpenAiCompatibleClient:
            return _FakeOpenAiCompatibleClient(api_key=api_key, base_url=base_url, dim=1024)

        result = fetch_voyage_embeddings(["a", "bb", "ccc"], api_key="k", client_factory=_factory)

        assert result.shape == (3, 1024)
        assert np.allclose(result[:, 0], [1.0, 2.0, 3.0])

    def test_raises_a_clear_error_if_a_returned_embedding_is_not_1024_dimensional(self):
        class _WrongDimEmbeddingsResource:
            def create(self, **kwargs: object) -> _FakeEmbeddingResponse:
                return _FakeEmbeddingResponse(data=[_FakeEmbedding([0.0] * 768)])

        class _WrongDimClient:
            def __init__(self, *, api_key: str, base_url: str) -> None:
                self.embeddings = _WrongDimEmbeddingsResource()

        with pytest.raises(RuntimeError, match="1024"):
            fetch_voyage_embeddings(["a"], api_key="k", client_factory=_WrongDimClient)
