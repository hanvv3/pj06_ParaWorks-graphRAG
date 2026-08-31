import pytest

from backend.app.rag.embeddings import (
    OpenAIEmbeddingConfig,
    OpenAIEmbeddingModel,
    openai_compatible_embedding_config,
    validate_query_embedding_vector,
)


class FakeEmbeddingResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class FakeHttpClient:
    def __init__(self, response_payload: dict) -> None:
        self.response_payload = response_payload
        self.requests: list[dict] = []

    def post(self, url: str, *, headers: dict, json: dict, timeout: float) -> FakeEmbeddingResponse:
        self.requests.append(
            {
                'url': url,
                'headers': headers,
                'json': json,
                'timeout': timeout,
            }
        )
        return FakeEmbeddingResponse(self.response_payload)


def test_openai_embedding_model_batches_inputs_and_tracks_usage() -> None:
    client = FakeHttpClient(
        {
            'data': [
                {'index': 1, 'embedding': [0.0, 1.0]},
                {'index': 0, 'embedding': [1.0, 0.0]},
            ],
            'usage': {'prompt_tokens': 12, 'total_tokens': 12},
        }
    )
    model = OpenAIEmbeddingModel(
        config=OpenAIEmbeddingConfig(
            api_key='test-key',
            model='text-embedding-3-small',
            dimensions=2,
            timeout_seconds=7.0,
        ),
        http_client=client,
    )

    result = model.embed_many(['첫 번째 문서', 'second document'])

    assert result.embeddings == [[1.0, 0.0], [0.0, 1.0]]
    assert result.prompt_tokens == 12
    assert result.total_tokens == 12
    assert result.request_count == 1
    assert client.requests == [
        {
            'url': 'https://api.openai.com/v1/embeddings',
            'headers': {
                'Authorization': 'Bearer test-key',
                'Content-Type': 'application/json',
            },
            'json': {
                'model': 'text-embedding-3-small',
                'input': ['첫 번째 문서', 'second document'],
                'encoding_format': 'float',
                'dimensions': 2,
            },
            'timeout': 7.0,
        }
    ]


def test_openai_compatible_embedding_config_accepts_azure_openai_alias_with_openai_key() -> None:
    config = openai_compatible_embedding_config(
        provider='azure_openai',
        api_key='openai-compatible-key',
        model='text-embedding-3-small',
        dimensions=1536,
    )

    assert config.api_key == 'openai-compatible-key'
    assert config.model == 'text-embedding-3-small'
    assert config.base_url == 'https://api.openai.com/v1'


def test_query_vector_carrier_is_float32_canonical_and_rejects_bool_or_zero() -> None:
    carrier = validate_query_embedding_vector(
        [0.1, -0.2],
        expected_dimensions=2,
    )
    assert carrier.coordinates == (
        0.10000000149011612,
        -0.20000000298023224,
    )
    with pytest.raises(ValueError, match='cosine-indexable'):
        validate_query_embedding_vector([True, 0.0], expected_dimensions=2)
    with pytest.raises(ValueError, match='cosine-indexable'):
        validate_query_embedding_vector([0.0, -0.0], expected_dimensions=2)
