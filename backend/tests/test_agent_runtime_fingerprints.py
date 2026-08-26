import unicodedata

import pytest

from backend.app.agent_runtime.fingerprints import (
    canonical_json_bytes,
    keyed_fingerprint,
)


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


def test_keyed_fingerprint_unambiguously_binds_both_versions() -> None:
    payload = {'source_refs': ['source-a:v1']}
    secret = b'test-only-fingerprint-secret'

    split_after_schema = keyed_fingerprint(
        payload,
        secret=secret,
        schema_version='alpha\nbeta',
        policy_version='gamma',
    )
    split_after_policy = keyed_fingerprint(
        payload,
        secret=secret,
        schema_version='alpha',
        policy_version='beta\ngamma',
    )
    next_schema = keyed_fingerprint(
        payload,
        secret=secret,
        schema_version='alpha:v2',
        policy_version='gamma',
    )

    assert split_after_schema != split_after_policy
    assert split_after_schema != next_schema


def test_canonical_json_rejects_recursive_containers_with_value_error() -> None:
    recursive: list[object] = []
    recursive.append(recursive)

    with pytest.raises(ValueError, match='nesting is too deep or cyclic'):
        canonical_json_bytes(recursive)


def test_canonical_json_rejects_excessive_nesting_with_value_error() -> None:
    nested: object = None
    for _ in range(2_000):
        nested = [nested]

    with pytest.raises(ValueError, match='nesting is too deep or cyclic'):
        canonical_json_bytes(nested)
