import hashlib
import hmac
import json
import math
import unicodedata

from backend.app.core.config import Settings

LOCAL_FINGERPRINT_SECRET = 'local-development-agent-runtime-fingerprint-secret'


def fingerprint_secret_bytes(settings: Settings) -> tuple[bytes, str]:
    secret = settings.agent_runtime_fingerprint_secret
    key_version = settings.agent_runtime_fingerprint_key_version.strip()
    if not secret or not key_version:
        raise ValueError('agent runtime fingerprint configuration is incomplete')
    if settings.paraworks_env == 'production' and secret == LOCAL_FINGERPRINT_SECRET:
        raise ValueError('production requires a dedicated fingerprint secret')
    return secret.encode('utf-8'), key_version


def _normalize_json_value(value: object) -> object:
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('fingerprint payload contains a non-finite float')
        return value
    if type(value) is str:
        return unicodedata.normalize('NFC', value)
    if type(value) is list:
        return [_normalize_json_value(item) for item in value]
    if type(value) is dict:
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError('fingerprint payload keys must be strings')
            normalized_key = unicodedata.normalize('NFC', key)
            if normalized_key in normalized:
                raise ValueError('fingerprint payload has duplicate normalized keys')
            normalized[normalized_key] = _normalize_json_value(item)
        return normalized
    raise ValueError('fingerprint payload must contain JSON-safe primitives')


def canonical_json_bytes(value: object) -> bytes:
    try:
        normalized = _normalize_json_value(value)
        return json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            separators=(',', ':'),
            sort_keys=True,
        ).encode('utf-8')
    except RecursionError:
        raise ValueError(
            'fingerprint payload nesting is too deep or cyclic'
        ) from None


def keyed_fingerprint(
    value: object,
    *,
    secret: bytes,
    schema_version: str,
    policy_version: str,
) -> str:
    if not secret or not schema_version.strip() or not policy_version.strip():
        raise ValueError('fingerprint secret and versions are required')
    payload = canonical_json_bytes({
        'domain': 'paraworks:keyed-fingerprint:v1',
        'policy_version': policy_version,
        'schema_version': schema_version,
        'value': value,
    })
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()
