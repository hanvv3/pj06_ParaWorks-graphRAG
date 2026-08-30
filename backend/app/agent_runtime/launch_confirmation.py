from __future__ import annotations

import base64
import hashlib
import hmac
import json
import unicodedata
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Final, TypeAlias

from backend.app.agent_runtime.fingerprints import fingerprint_secret_bytes
from backend.app.agent_runtime.review_v2_preflight import PreparedReviewRequestV21
from backend.app.core.config import Settings

LaunchScalar: TypeAlias = str | int

_SNAPSHOT_KEYS: Final[frozenset[str]] = frozenset({
    'v', 'g', 'm',
    'sh', 'ah', 'ph', 'ih', 'eh',
    'vp', 'vm', 'vr', 'vs', 'pp', 'pv', 'cv', 'ps',
    'xv', 'xp', 'xm', 'xr', 'xt', 'xn',
    'xc', 'xi', 'xo', 'xk',
    'xe', 'xen', 'xrp', 'xfs', 'xip', 'xop',
    'ep', 'es',
    'rc', 'rq', 'ap', 'rg',
    'kv', 'km',
    'te', 'en', 'rp', 'fs', 'it', 'ot',
    'mb', 'mc', 'mw', 'ma',
    'pt', 'sw', 'ls', 'cg',
    'ip', 'op', 'bl', 'ec', 'vc', 'tc',
})
_TOKEN_KEYS: Final[frozenset[str]] = _SNAPSHOT_KEYS | {'iat', 'exp'}


class LaunchConfirmationError(ValueError):  # noqa: N818 - bounded domain error
    def __init__(self) -> None:
        super().__init__('cost_preview_changed')


class LaunchConfirmationCodec:
    """Frozen compact HMAC codec for one zero-call V2.1 launch preview."""

    exact_keys = _TOKEN_KEYS

    def __init__(
        self,
        *,
        settings: Settings,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._settings = settings
        self._now = now or (lambda: datetime.now(UTC))

    def issue(self, snapshot: Mapping[str, LaunchScalar]) -> str:
        try:
            values = dict(snapshot)
            if set(values) != _SNAPSHOT_KEYS:
                raise ValueError
            if any(
                isinstance(value, bool) or not isinstance(value, (str, int))
                for value in values.values()
            ):
                raise ValueError
            issued_at = int(self._utc_now().timestamp())
            values['iat'] = issued_at
            values['exp'] = (
                issued_at
                + self._settings.auto_review_launch_confirmation_ttl_seconds
            )
            body = _canonical_json(values)
            secret, _ = fingerprint_secret_bytes(self._settings)
            signature = hmac.new(secret, body, hashlib.sha256).digest()
            token = f'{_b64(body)}.{_b64(signature)}'
            if len(token) > 2048:
                raise ValueError
            return token
        except LaunchConfirmationError:
            raise
        except Exception:
            raise LaunchConfirmationError() from None

    def verify(
        self,
        token: str,
        *,
        expected: Mapping[str, LaunchScalar],
    ) -> dict[str, LaunchScalar]:
        try:
            if not isinstance(token, str) or len(token) > 2048:
                raise ValueError
            encoded_body, encoded_signature = token.split('.', maxsplit=1)
            body = _unb64(encoded_body)
            signature = _unb64(encoded_signature)
            secret, _ = fingerprint_secret_bytes(self._settings)
            expected_signature = hmac.new(secret, body, hashlib.sha256).digest()
            if not hmac.compare_digest(signature, expected_signature):
                raise ValueError
            payload = json.loads(body.decode('utf-8'))
            if not isinstance(payload, dict) or set(payload) != _TOKEN_KEYS:
                raise ValueError
            snapshot = {key: payload[key] for key in _SNAPSHOT_KEYS}
            if set(expected) != _SNAPSHOT_KEYS or snapshot != dict(expected):
                raise ValueError
            now = int(self._utc_now().timestamp())
            if (
                not isinstance(payload['iat'], int)
                or not isinstance(payload['exp'], int)
                or payload['iat'] > now
                or now >= payload['exp']
                or payload['exp'] - payload['iat']
                != self._settings.auto_review_launch_confirmation_ttl_seconds
            ):
                raise ValueError
            return payload
        except LaunchConfirmationError:
            raise
        except Exception:
            raise LaunchConfirmationError() from None

    def _utc_now(self) -> datetime:
        value = self._now()
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


def build_v21_launch_snapshot(
    *,
    prepared: PreparedReviewRequestV21,
    security_scope_hmac: str,
    actor_subject_hmac: str,
    owner_permission_hmac: str,
) -> dict[str, LaunchScalar]:
    config = prepared.config
    if not config.extraction_plan_identities:
        raise LaunchConfirmationError()
    extraction = config.extraction_plan_identities[0]
    common_fields = (
        'provider',
        'model',
        'reasoning_effort',
        'route_version',
        'cost_policy_version',
        'token_estimator_version',
        'tokenizer_encoding',
        'reply_priming_tokens',
        'framing_safety_tokens',
        'max_input_chars',
        'max_input_tokens',
        'max_output_tokens',
        'max_candidates',
        'max_provider_attempts',
        'input_usd_per_1m',
        'output_usd_per_1m',
        'timing',
    )
    if any(field not in extraction for field in common_fields):
        raise LaunchConfirmationError()
    common = {field: extraction[field] for field in common_fields}
    if any(
        any(identity.get(field) != value for field, value in common.items())
        for identity in config.extraction_plan_identities[1:]
    ):
        raise LaunchConfirmationError()
    timing = tuple(common['timing'])
    if timing != (
        config.provider_timeout_seconds,
        config.provider_send_start_window_seconds,
        config.provider_attempt_lease_seconds,
        config.provider_commit_grace_seconds,
    ):
        raise LaunchConfirmationError()
    values: dict[str, LaunchScalar] = {
        'v': 1,
        'g': prepared.graph_version,
        'm': config.configured_auto_review_mode,
        'sh': security_scope_hmac,
        'ah': actor_subject_hmac,
        'ph': owner_permission_hmac,
        'ih': prepared.input_hash,
        'eh': prepared.evidence_version_hash,
        'vp': config.validator_provider,
        'vm': config.validator_model,
        'vr': config.validator_reasoning_effort,
        'vs': config.validator_output_contract_version,
        'pp': config.validator_prompt_version,
        'pv': config.policy_version,
        'cv': config.cost_policy_version,
        'ps': config.validation_provider_safety_state_version,
        'xv': str(common['cost_policy_version']),
        'xp': str(common['provider']),
        'xm': str(common['model']),
        'xr': str(common['reasoning_effort']),
        'xt': str(common['route_version']),
        'xn': len(config.extraction_plan_identities),
        'xc': int(common['max_input_chars']),
        'xi': int(common['max_input_tokens']),
        'xo': int(common['max_output_tokens']),
        'xk': int(common['max_candidates']),
        'xe': str(common['token_estimator_version']),
        'xen': str(common['tokenizer_encoding']),
        'xrp': int(common['reply_priming_tokens']),
        'xfs': int(common['framing_safety_tokens']),
        'xip': _decimal_text(common['input_usd_per_1m']),
        'xop': _decimal_text(common['output_usd_per_1m']),
        'ep': config.extraction_plan_set_hmac,
        'es': config.extraction_provider_safety_snapshot_set_hmac,
        'rc': config.rollout_control_epoch,
        'rq': config.enforce_percentage,
        'ap': config.authorized_percentage_at_launch,
        'rg': config.rollout_authorization_generation,
        'kv': config.fingerprint_key_version,
        'km': config.fingerprint_key_material_verifier,
        'te': config.token_estimator_version,
        'en': config.tokenizer_encoding,
        'rp': config.reply_priming_tokens,
        'fs': config.framing_safety_tokens,
        'it': config.max_input_tokens_per_batch,
        'ot': config.max_output_tokens_per_batch,
        'mb': config.max_validation_batches_per_workflow,
        'mc': config.max_validation_candidates_per_batch,
        'mw': config.max_validation_candidates_per_workflow,
        'ma': config.max_provider_attempts,
        'pt': config.provider_timeout_seconds,
        'sw': config.provider_send_start_window_seconds,
        'ls': config.provider_attempt_lease_seconds,
        'cg': config.provider_commit_grace_seconds,
        'ip': format(config.validator_input_usd_per_1m, 'f'),
        'op': format(config.validator_output_usd_per_1m, 'f'),
        'bl': format(config.total_budget_limit_usd, 'f'),
        'ec': format(config.confirmed_extraction_cost_ceiling_usd, 'f'),
        'vc': format(config.confirmed_validation_cost_ceiling_usd, 'f'),
        'tc': format(config.confirmed_total_cost_ceiling_usd, 'f'),
    }
    if set(values) != _SNAPSHOT_KEYS:
        raise LaunchConfirmationError()
    return values


def _decimal_text(value: object) -> str:
    return format(value, 'f') if hasattr(value, 'as_tuple') else str(value)


def _canonical_json(value: Mapping[str, LaunchScalar]) -> bytes:
    return unicodedata.normalize(
        'NFC',
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(',', ':'),
        ),
    ).encode('utf-8')


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b'=').decode('ascii')


def _unb64(value: str) -> bytes:
    if not value or any(character.isspace() for character in value):
        raise ValueError
    return base64.urlsafe_b64decode(value + '=' * (-len(value) % 4))
