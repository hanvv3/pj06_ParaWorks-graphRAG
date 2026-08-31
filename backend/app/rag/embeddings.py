import hashlib
import math
import re
import struct
from dataclasses import dataclass
from typing import Protocol

import httpx

from backend.app.rag.vector_validation import (
    CanonicalFloat32Vector,
    CosineIndexableVectorValidator,
)


@dataclass(frozen=True, slots=True)
class ValidatedQueryEmbeddingVector:
    """Canonical query vector; provider and storage metadata are not authority."""

    coordinates: CanonicalFloat32Vector
    canonical_big_endian_float32_sha256: str
    cosine_indexability_policy_version: str


def validate_query_embedding_vector(
    value: object,
    *,
    expected_dimensions: int = 1536,
) -> ValidatedQueryEmbeddingVector:
    if not isinstance(value, list):
        raise ValueError('query embedding must be a JSON array')
    coordinates = CosineIndexableVectorValidator().validate(
        value,
        expected_dimensions=expected_dimensions,
    )
    payload = b'paraworks:pgvector-float32:v1\x00' + b''.join(
        struct.pack('>f', coordinate) for coordinate in coordinates
    )
    return ValidatedQueryEmbeddingVector(
        coordinates=coordinates,
        canonical_big_endian_float32_sha256=hashlib.sha256(payload).hexdigest(),
        cosine_indexability_policy_version='pgvector-cosine-indexable:v1',
    )


@dataclass(frozen=True)
class EmbeddingBatchResult:
    embeddings: list[list[float]]
    prompt_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0


class EmbeddingModel(Protocol):
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_many(self, texts: list[str]) -> EmbeddingBatchResult:
        raise NotImplementedError


class DeterministicHashEmbeddingModel:
    def __init__(self, dimensions: int = 16) -> None:
        if dimensions <= 0:
            raise ValueError('dimensions must be greater than zero')
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokens(text):
            digest = hashlib.sha256(token.encode('utf-8')).digest()
            index = int.from_bytes(digest[:4], byteorder='big') % self.dimensions
            sign = -1.0 if digest[4] % 2 else 1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            return vector
        return [value / norm for value in vector]

    def embed_many(self, texts: list[str]) -> EmbeddingBatchResult:
        return EmbeddingBatchResult(
            embeddings=[self.embed(text) for text in texts],
            request_count=1 if texts else 0,
        )


@dataclass(frozen=True)
class OpenAIEmbeddingConfig:
    api_key: str
    model: str = 'text-embedding-3-small'
    dimensions: int | None = 1536
    base_url: str = 'https://api.openai.com/v1'
    timeout_seconds: float = 30.0


def openai_compatible_embedding_config(
    *,
    provider: str,
    api_key: str,
    model: str = 'text-embedding-3-small',
    dimensions: int | None = 1536,
    base_url: str | None = None,
    timeout_seconds: float = 30.0,
) -> OpenAIEmbeddingConfig:
    normalized_provider = provider.strip().lower()
    if normalized_provider not in {'openai', 'azure_openai'}:
        raise ValueError(f'Unsupported OpenAI-compatible embedding provider: {provider}')
    return OpenAIEmbeddingConfig(
        api_key=api_key,
        model=model,
        dimensions=dimensions,
        base_url=base_url or 'https://api.openai.com/v1',
        timeout_seconds=timeout_seconds,
    )


class OpenAIEmbeddingModel:
    def __init__(
        self,
        *,
        config: OpenAIEmbeddingConfig,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self.dimensions = config.dimensions or 0
        self._http_client = http_client or httpx.Client()

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text]).embeddings[0]

    def embed_many(self, texts: list[str]) -> EmbeddingBatchResult:
        if not texts:
            return EmbeddingBatchResult(embeddings=[], request_count=0)

        body: dict[str, object] = {
            'model': self.config.model,
            'input': texts,
            'encoding_format': 'float',
        }
        if self.config.dimensions is not None:
            body['dimensions'] = self.config.dimensions

        response = self._http_client.post(
            f'{self.config.base_url.rstrip("/")}/embeddings',
            headers={
                'Authorization': f'Bearer {self.config.api_key}',
                'Content-Type': 'application/json',
            },
            json=body,
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        data = sorted(payload.get('data', []), key=lambda item: int(item['index']))
        usage = payload.get('usage', {})
        return EmbeddingBatchResult(
            embeddings=[[float(value) for value in item['embedding']] for item in data],
            prompt_tokens=int(usage.get('prompt_tokens', 0)),
            total_tokens=int(usage.get('total_tokens', 0)),
            request_count=1,
        )


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r'[a-zA-Z0-9가-힣]+', text.lower()) if len(token) >= 2]
