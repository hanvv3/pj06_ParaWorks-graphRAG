import unicodedata

from backend.app.agent_runtime.fingerprints import keyed_fingerprint


def test_keyed_fingerprint_is_stable_for_key_order_and_unicode_form() -> None:
    secret = b'test-only-fingerprint-secret'
    composed = '담당자 김하나'
    decomposed = unicodedata.normalize('NFD', composed)

    left = keyed_fingerprint(
        {'name': composed, 'agents': ['mail_document_agent', 'memory_extraction_agent']},
        secret=secret,
        schema_version='review-input:v1',
        policy_version='ranked:v1',
    )
    right = keyed_fingerprint(
        {'agents': ['mail_document_agent', 'memory_extraction_agent'], 'name': decomposed},
        secret=secret,
        schema_version='review-input:v1',
        policy_version='ranked:v1',
    )

    assert left == right
    assert len(left) == 64
    assert composed not in left


def test_keyed_fingerprint_changes_for_list_order_policy_or_secret() -> None:
    payload = {'source_refs': ['source-a:v1', 'source-b:v2']}
    baseline = keyed_fingerprint(
        payload,
        secret=b'secret-a',
        schema_version='review-input:v1',
        policy_version='ranked:v1',
    )

    assert baseline != keyed_fingerprint(
        {'source_refs': list(reversed(payload['source_refs']))},
        secret=b'secret-a',
        schema_version='review-input:v1',
        policy_version='ranked:v1',
    )
    assert baseline != keyed_fingerprint(
        payload,
        secret=b'secret-a',
        schema_version='review-input:v1',
        policy_version='ranked:v2',
    )
    assert baseline != keyed_fingerprint(
        payload,
        secret=b'secret-b',
        schema_version='review-input:v1',
        policy_version='ranked:v1',
    )
