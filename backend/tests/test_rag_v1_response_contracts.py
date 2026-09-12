import importlib

import pytest
from pydantic import TypeAdapter, ValidationError

ASK_KEYS = {
    'agent_name',
    'prompt_version',
    'question',
    'answer',
    'source_ids',
    'source_links',
    'source_snippets',
    'citations',
    'permission_level',
    'hidden_match_count',
    'permission_notice',
    'agent_run_id',
    'cache_key',
    'model_name',
    'estimated_cost_usd',
    'token_usage',
}
CITATION_KEYS = {
    'source_id',
    'source_url',
    'source_type',
    'permission_level',
    'source_snippet',
    'relevance_score',
    'matched_terms',
}
RESULT_KEYS = CITATION_KEYS | {
    'id',
    'text',
    'citation',
    'parser_status',
    'parser_status_reason',
    'revision_id',
}
ERROR_CODES = {
    'input_safety_blocked',
    'input_scanner_unavailable',
    'permission_denied',
    'budget_exceeded',
    'runtime_version_unavailable',
    'retriever_not_configured',
    'retriever_unavailable',
    'model_unavailable',
    'provider_safety_unavailable',
    'provider_response_identity_invalid',
    'provider_usage_overrun',
    'provider_embedding_payload_invalid',
    'model_provider_failed',
    'structured_output_invalid',
    'citation_validation_failed',
    'persistence_failed',
    'unexpected_internal_error',
}


def test_exact_response_models_are_registered_in_openapi(client):
    schemas = client.get('/openapi.json').json()['components']['schemas']
    assert 'AskV1Projection' in schemas
    assert set(schemas['AskV1Projection']['properties']) == ASK_KEYS
    assert set(schemas['AskV1Projection']['required']) == ASK_KEYS
    assert set(schemas['RagCitationResponse']['properties']) == CITATION_KEYS
    assert set(schemas['RagCitationResponse']['required']) == CITATION_KEYS
    assert set(schemas['SearchResultResponse']['required']) == RESULT_KEYS
    assert schemas['RagCitationResponse']['properties']['source_url'] == {
        'title': 'Source Url',
        'type': 'string',
    }
    assert schemas['AskV1Projection']['properties']['citations']['type'] == 'array'
    assert 'prefixItems' not in schemas['AskV1Projection']['properties']['citations']
    assert (
        set(schemas['RagPublicErrorDetail']['properties']['code']['enum'])
        == ERROR_CODES
    )
    responses = client.get('/openapi.json').json()['paths']['/api/v1/ask']['post'][
        'responses'
    ]
    assert 'anyOf' in responses['422']['content']['application/json']['schema']


def test_error_decoder_rejects_unknown_codes_and_extra_fields():
    assert importlib.util.find_spec('backend.app.schemas.rag') is not None
    rag = importlib.import_module('backend.app.schemas.rag')
    decoder = TypeAdapter(rag.RagPublicErrorResponse)
    for code in ERROR_CODES:
        assert decoder.validate_python({'detail': {'code': code}}).model_dump() == {
            'detail': {'code': code}
        }
    for body in (
        {'detail': {'code': 'unknown'}},
        {'detail': {'code': 'budget_exceeded', 'trace': 'secret'}},
        {'detail': {'code': 'budget_exceeded'}, 'trace': 'secret'},
    ):
        with pytest.raises(ValidationError):
            decoder.validate_python(body)


@pytest.mark.parametrize('endpoint,field', [('ask', 'question'), ('search', 'query')])
def test_fastapi_validation_body_is_preserved(client, endpoint, field):
    response = client.post(f'/api/v1/{endpoint}', json={field: ''})
    assert response.status_code == 422
    assert isinstance(response.json()['detail'], list)


@pytest.mark.parametrize('endpoint,field', [('ask', 'question'), ('search', 'query')])
@pytest.mark.parametrize('value', ['a\u0000b', '\ud800', '\udfff'])
def test_invalid_unicode_keeps_fastapi_422_detail_array(client, endpoint, field, value):
    import json

    response = client.post(
        f'/api/v1/{endpoint}',
        content=json.dumps({field: value}),
        headers={'Content-Type': 'application/json'},
    )
    assert response.status_code == 422
    assert isinstance(response.json()['detail'], list)


def test_legacy_exact_recursive_keys_and_required_nulls(client, db_session):
    from backend.tests.test_rag_orchestrator_service import seed_chunk

    seed_chunk(
        db_session, 'gmail', 'gmail-contract', 'Contract observation', 'internal'
    )
    ask = client.post('/api/v1/ask', json={'question': 'Contract'}).json()
    search = client.post('/api/v1/search', json={'query': 'Contract'}).json()
    assert set(ask) == ASK_KEYS
    assert set(ask['token_usage']) == {'input_tokens', 'output_tokens', 'total_tokens'}
    assert set(ask['citations'][0]) == CITATION_KEYS
    assert set(search) == {
        'retrieval_backend',
        'cost_policy',
        'hidden_match_count',
        'results',
    }
    assert set(search['results'][0]) == RESULT_KEYS
    assert set(search['results'][0]['citation']) == CITATION_KEYS
    assert search['results'][0]['parser_status_reason'] is None


def test_dto_rejects_extra_null_url_and_nested_mutation():
    assert importlib.util.find_spec('backend.app.schemas.rag') is not None
    rag = importlib.import_module('backend.app.schemas.rag')
    citation = {
        'source_id': 'gmail:1',
        'source_url': 'https://example.test/1',
        'source_type': None,
        'permission_level': 'internal',
        'source_snippet': 'proof',
        'relevance_score': 1.0,
        'matched_terms': ['proof'],
    }
    dto = rag.RagCitationResponse(**citation)
    assert dto.model_dump(mode='json') == citation
    with pytest.raises((AttributeError, TypeError)):
        dto.matched_terms.append('changed')
    for mutation in ({'source_url': None}, {'internal_identity': 'secret'}):
        with pytest.raises(ValidationError):
            rag.RagCitationResponse(**(citation | mutation))
