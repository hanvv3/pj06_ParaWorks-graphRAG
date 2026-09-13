from __future__ import annotations

import hashlib
import hmac
import json
from copy import copy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from pydantic import SecretStr

from backend.app.rag import release_review as review
from backend.tests.test_rag_live_gate_preview import SECRET, utf8


def encoded(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(',', ':')
    ).encode()


ROLES = ('reviewer_a', 'reviewer_b', 'adjudicator_c')
NOW = datetime(2026, 9, 13, 1, 2, 3, tzinfo=UTC)
REDIRECT = 'http://127.0.0.1:8765/rag-live-gate/reviewer/callback'


def subject_digest(subject):
    return hmac.new(
        SECRET,
        encoded(
            {
                'domain': 'paraworks:keyed-fingerprint:v1',
                'schema_version': 'rag-live-authenticated-subject:v1',
                'policy_version': 'rag-live-quality-rubric:v1',
                'value': {
                    'authenticated_subject_bytes': utf8(subject),
                    'issuer_bytes': utf8('https://accounts.google.com'),
                },
            }
        ),
        hashlib.sha256,
    ).hexdigest()


@pytest.fixture
def reviewer_harness():
    return make_reviewer_harness


def make_reviewer_harness():
    # A missing implementation must fail as an assertion, not a collection error.
    cls = getattr(review, 'FreshGoogleReviewerVerifier', None)
    assert cls is not None, 'Task24-B fresh reviewer verifier is missing'
    users = {
        role: SimpleNamespace(id=i, external_id=f'google-subject-{i}', status='active')
        for i, role in enumerate(ROLES, 1)
    }
    state = SimpleNamespace(
        now=NOW, users=users, calls=[], response={}, selected=ROLES[0]
    )

    def handle(request):
        state.calls.append(request)
        if request.url == 'https://oauth2.googleapis.com/token':
            if state.response.get('raise'):
                raise ValueError('sensitive-provider-exception')
            return httpx.Response(
                200,
                json=state.response.get(
                    'token',
                    {
                        'access_token': 'test-ephemeral-access',
                        'token_type': 'Bearer',
                        'expires_in': 3599,
                    },
                ),
            )
        assert request.url == 'https://openidconnect.googleapis.com/v1/userinfo'
        assert request.headers['Authorization'] == 'Bearer test-ephemeral-access'
        return httpx.Response(
            200,
            json=state.response.get(
                'userinfo',
                {
                    'sub': users[state.selected].external_id,
                    'email': 'irrelevant@example.invalid',
                    'email_verified': True,
                },
            ),
        )

    client = httpx.Client(transport=httpx.MockTransport(handle), follow_redirects=False)
    verifier = cls(
        settings=SimpleNamespace(
            google_client_id='release-client',
            google_client_secret='test-client-secret',
            google_identity_state_secret='test-state-secret',
        ),
        redirect_uri=REDIRECT,
        selected_auth_user_ids={role: user.id for role, user in users.items()},
        auth_user_reader=lambda user_id: next(
            u for u in users.values() if u.id == user_id
        ),
        identity_secret=SECRET,
        http_client=client,
        clock=lambda: state.now,
    )
    state.verifier = verifier
    return state


def begin(harness, role='reviewer_a'):
    harness.selected = role
    challenge = harness.verifier.begin(role=role)
    params = parse_qs(urlparse(challenge.authorization_url.get_secret_value()).query)
    return challenge, params


def complete(harness, challenge, params, **overrides):
    arguments = {
        'role': challenge.role,
        'code': SecretStr(f'test-ephemeral-code-{challenge.role}'),
        'signed_state': SecretStr(params['state'][0]),
        'expected_challenge_hmac': challenge.challenge_hmac,
    }
    arguments.update(overrides)
    return harness.verifier.complete(**arguments)


def test_fresh_pkce_proof_has_fixed_endpoints_and_only_subject_hmac(
    reviewer_harness, caplog
):
    h = reviewer_harness()
    challenge, params = begin(h)
    assert params['client_id'] == ['release-client']
    assert params['redirect_uri'] == [REDIRECT]
    assert params['response_type'] == ['code']
    assert params['code_challenge_method'] == ['S256']
    proof = complete(h, challenge, params)
    exchange = parse_qs(h.calls[0].content.decode())
    assert exchange['redirect_uri'] == [REDIRECT]
    assert exchange['client_id'] == ['release-client']
    assert exchange['code'] == ['test-ephemeral-code-reviewer_a']
    assert exchange['grant_type'] == ['authorization_code']
    import base64

    assert (
        base64.urlsafe_b64encode(
            hashlib.sha256(exchange['code_verifier'][0].encode()).digest()
        )
        .decode()
        .rstrip('=')
        == params['code_challenge'][0]
    )
    assert proof.reviewer_subject_hmac == subject_digest('google-subject-1')
    assert proof.role == 'reviewer_a' and proof.auth_user_id == 1
    assert proof.issuer == 'https://accounts.google.com'
    assert proof.challenge_hmac == challenge.challenge_hmac
    assert proof.authenticated_at_utc == NOW
    for secret in (
        'test-ephemeral-code',
        'test-ephemeral-access',
        params['state'][0],
        'google-subject-1',
    ):
        assert (
            secret not in repr(challenge) + repr(proof) + repr(h.verifier) + caplog.text
        )
    with pytest.raises(review.LiveGatePreviewError):
        complete(h, challenge, params)
    assert len(h.calls) == 2


@pytest.mark.parametrize(
    'attack',
    [
        'state',
        'challenge',
        'role',
        'expired',
        'future',
        'code_type',
        'state_type',
        'empty_code',
        'inactive',
        'subject_drift',
    ],
)
def test_invalid_callback_refuses_before_exchange(reviewer_harness, attack):
    h = reviewer_harness()
    challenge, params = begin(h)
    args = {}
    if attack == 'state':
        args['signed_state'] = SecretStr(params['state'][0] + 'x')
    if attack == 'challenge':
        args['expected_challenge_hmac'] = 'f' * 64
    if attack == 'role':
        args['role'] = 'reviewer_b'
    if attack == 'expired':
        h.now = challenge.expires_at_utc
    if attack == 'future':
        h.now = challenge.issued_at_utc - timedelta(microseconds=1)
    if attack == 'code_type':
        args['code'] = 'test-ephemeral-code'
    if attack == 'state_type':
        args['signed_state'] = params['state'][0]
    if attack == 'empty_code':
        args['code'] = SecretStr(' ')
    if attack == 'inactive':
        h.users['reviewer_a'].status = 'inactive'
    if attack == 'subject_drift':
        h.users['reviewer_a'].external_id = 'changed-subject'
    with pytest.raises(review.LiveGatePreviewError):
        complete(h, challenge, params, **args)
    assert h.calls == []


@pytest.mark.parametrize(
    'response',
    [
        {'token': {}},
        {'token': {'access_token': ' ', 'token_type': 'Bearer'}},
        {'token': {'access_token': 42, 'token_type': 'Bearer'}},
        {'token': {'access_token': 'test-ephemeral-access', 'token_type': 'Other'}},
        {'userinfo': {'email': 'google-subject-1'}},
        {'userinfo': {'sub': ' '}},
        {'userinfo': {'sub': 1}},
        {'userinfo': {'sub': 'another-subject'}},
        {'userinfo': {'sub': 'google-subject-1', 'iss': 'https://unexpected.invalid'}},
        {'raise': True},
    ],
)
def test_invalid_google_response_is_sanitized_and_challenge_consumed(
    reviewer_harness, response
):
    h = reviewer_harness()
    challenge, params = begin(h)
    h.response = response
    with pytest.raises(review.LiveGatePreviewError) as exc:
        complete(h, challenge, params)
    assert 'sensitive' not in str(exc.value)
    attempted = len(h.calls)
    h.response = {}
    with pytest.raises(review.LiveGatePreviewError):
        complete(h, challenge, params)
    assert len(h.calls) == attempted


def authenticated_roster(h):
    proofs = []
    for role in ROLES:
        challenge, params = begin(h, role)
        proofs.append(complete(h, challenge, params))
    return tuple(proofs)


def test_exact_roster_hmac_and_unissued_role_mutation_refusal(reviewer_harness):
    h = reviewer_harness()
    proofs = authenticated_roster(h)
    payload = {
        f'{role}_subject_hmac': subject_digest(f'google-subject-{i}')
        for i, role in enumerate(ROLES, 1)
    } | {'roster_version': 'rag-live-reviewer-roster:v1'}
    expected = hmac.new(
        SECRET,
        encoded(
            {
                'domain': 'paraworks:keyed-fingerprint:v1',
                'schema_version': 'rag-live-reviewer-roster:v1',
                'policy_version': 'rag-live-quality-rubric:v1',
                'value': payload,
            }
        ),
        hashlib.sha256,
    ).hexdigest()
    assert h.verifier.reviewer_roster_hmac(proofs) == expected
    for invalid in (
        (proofs[1], proofs[0], proofs[2]),
        (replace(proofs[0], role='reviewer_b'), proofs[1], proofs[2]),
        (copy(proofs[0]), proofs[1], proofs[2]),
        (proofs[0], proofs[0], proofs[2]),
    ):
        with pytest.raises(review.LiveGatePreviewError):
            h.verifier.reviewer_roster_hmac(invalid)
    object.__setattr__(proofs[0], 'reviewer_subject_hmac', 'a' * 64)
    with pytest.raises(review.LiveGatePreviewError):
        h.verifier.reviewer_roster_hmac(proofs)


def test_subject_reassigned_between_fresh_roles_cannot_form_roster(reviewer_harness):
    h = reviewer_harness()
    challenge, params = begin(h)
    first = complete(h, challenge, params)
    h.users['reviewer_b'].external_id = h.users['reviewer_a'].external_id
    with pytest.raises(review.LiveGatePreviewError):
        challenge, params = begin(h, 'reviewer_b')
        complete(h, challenge, params)
    assert first.reviewer_subject_hmac == subject_digest('google-subject-1')


def test_code_reused_under_another_fresh_challenge_is_not_exchanged_twice(
    reviewer_harness,
):
    h = reviewer_harness()
    first, params = begin(h)
    complete(h, first, params)
    second, new_params = begin(h, 'reviewer_b')
    with pytest.raises(review.LiveGatePreviewError):
        complete(
            h, second, new_params, code=SecretStr('test-ephemeral-code-reviewer_a')
        )
    assert len(h.calls) == 2


@pytest.mark.parametrize('field', ['code_verifier', 'nonce'])
def test_signed_state_with_changed_pkce_or_nonce_cannot_replace_exact_challenge(
    reviewer_harness, field
):
    from backend.app.auth.google_identity import GoogleIdentityStateSigner

    h = reviewer_harness()
    challenge, params = begin(h)
    signer = GoogleIdentityStateSigner('test-state-secret')
    original = signer.validate(params['state'][0])
    replacement = signer.create(
        nonce='changed' if field == 'nonce' else original.nonce,
        code_verifier='changed' if field == 'code_verifier' else original.code_verifier,
    )
    with pytest.raises(review.LiveGatePreviewError):
        complete(h, challenge, params, signed_state=SecretStr(replacement))
    assert h.calls == []


def test_roster_rejects_user_reassignment_and_new_challenge_invalidates_old_proof(
    reviewer_harness,
):
    h = reviewer_harness()
    proofs = authenticated_roster(h)
    h.users['reviewer_a'].external_id = 'reassigned'
    with pytest.raises(review.LiveGatePreviewError):
        h.verifier.reviewer_roster_hmac(proofs)
    h.users['reviewer_a'].external_id = 'google-subject-1'
    begin(h)
    with pytest.raises(review.LiveGatePreviewError):
        h.verifier.reviewer_roster_hmac(proofs)
