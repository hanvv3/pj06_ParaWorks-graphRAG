from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Literal, TypeAlias

from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.core.config import Settings
from backend.app.knowledge.promotion import build_promotion_preview
from backend.app.models import ReviewItem
from backend.app.schemas.auto_review import AUTO_REVIEW_NORMALIZATION_SCHEMA_VERSION

KnowledgeEffectType: TypeAlias = Literal[
    'timeline_event',
    'history_event',
    'decision_record',
    'todo',
]
ReusableKnowledgeType: TypeAlias = Literal['timeline_event', 'history_event']

_SUBSTANTIVE_FIELDS: dict[KnowledgeEffectType, tuple[str, ...]] = {
    'timeline_event': ('title', 'result_summary'),
    'history_event': ('title', 'reason'),
    'decision_record': ('title', 'decision_summary'),
    'todo': ('title', 'assignee', 'due_date', 'priority', 'priority_reason'),
}
_OPTIONAL_FIELDS = frozenset({'assignee', 'due_date'})
_WHITESPACE = re.compile(r'\s+', re.UNICODE)


def normalize_trusted_text(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError('trusted claim fields must be strings')
    return _WHITESPACE.sub(' ', unicodedata.normalize('NFC', value)).strip()


def normalized_claim_fingerprint(
    *, item: ReviewItem, security_scope_id: str, settings: Settings
) -> str:
    preview = build_promotion_preview(item)
    target_type = preview.get('target_type')
    if target_type not in _SUBSTANTIVE_FIELDS:
        raise ValueError('review item does not have a trusted knowledge target')
    normalized = preview.get('normalized_payload')
    if not isinstance(normalized, dict):
        raise ValueError('promotion preview payload is invalid')
    fields: dict[str, str | None] = {
        key: normalized.get(key) for key in _SUBSTANTIVE_FIELDS[target_type]
    }
    if target_type == 'todo':
        fields['assignee'] = _optional_item_string(item, 'assignee')
        fields['due_date'] = _optional_item_string(item, 'due_date')
    return promoted_effect_fingerprint(
        knowledge_type=target_type,
        normalized_persisted_fields=fields,
        project_key=_optional_item_string(item, 'project_key'),
        security_scope_id=security_scope_id,
        settings=settings,
    )


def promoted_effect_fingerprint(
    *,
    knowledge_type: KnowledgeEffectType,
    normalized_persisted_fields: Mapping[str, str | None],
    project_key: str | None,
    security_scope_id: str,
    settings: Settings,
) -> str:
    expected = _SUBSTANTIVE_FIELDS.get(knowledge_type)
    if expected is None:
        raise ValueError('trusted knowledge type is unsupported')
    if set(normalized_persisted_fields) != set(expected):
        raise ValueError('trusted claim fields do not match the target schema')
    scope = normalize_trusted_text(security_scope_id)
    if not scope:
        raise ValueError('security scope is required')
    fields: dict[str, str | None] = {}
    for field in expected:
        raw = normalized_persisted_fields[field]
        if raw is None:
            if field not in _OPTIONAL_FIELDS:
                raise ValueError('trusted claim is missing a substantive field')
            fields[field] = None
            continue
        normalized = normalize_trusted_text(raw)
        if not normalized and field not in _OPTIONAL_FIELDS:
            raise ValueError('trusted claim has an empty substantive field')
        fields[field] = normalized or None
    secret, key_version = fingerprint_secret_bytes(settings)
    return keyed_fingerprint(
        {
            'fingerprint_key_version': key_version,
            'knowledge_type': knowledge_type,
            'normalization_version': AUTO_REVIEW_NORMALIZATION_SCHEMA_VERSION,
            'project_key': _normalize_optional(project_key),
            'security_scope_id': scope,
            'substantive_fields': fields,
        },
        secret=secret,
        schema_version=AUTO_REVIEW_NORMALIZATION_SCHEMA_VERSION,
        policy_version='trusted-claim-fingerprint:v1',
    )


def trusted_title_collision_bucket(
    *,
    item_type: ReusableKnowledgeType,
    normalized_title: str,
    settings: Settings,
) -> str:
    if item_type not in {'timeline_event', 'history_event'}:
        raise ValueError('collision buckets support Timeline and History only')
    title = normalize_trusted_text(normalized_title)
    if not title:
        raise ValueError('collision title is required')
    secret, key_version = fingerprint_secret_bytes(settings)
    return keyed_fingerprint(
        {
            'fingerprint_key_version': key_version,
            'knowledge_type': item_type,
            'normalization_version': AUTO_REVIEW_NORMALIZATION_SCHEMA_VERSION,
            'normalized_title': title,
        },
        secret=secret,
        schema_version='trusted-title-collision-bucket:v1',
        policy_version='trusted-title-collision-bucket:v1',
    )


def trusted_project_scope_fingerprint(
    *, project_key: str | None, settings: Settings
) -> str:
    secret, key_version = fingerprint_secret_bytes(settings)
    return keyed_fingerprint(
        {
            'fingerprint_key_version': key_version,
            'project_key': _normalize_optional(project_key),
        },
        secret=secret,
        schema_version='trusted-project-scope:v1',
        policy_version='trusted-project-scope:v1',
    )


def _normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = normalize_trusted_text(value)
    return normalized or None


def _optional_item_string(item: ReviewItem, key: str) -> str | None:
    value = (item.payload or {}).get(key)
    return value if isinstance(value, str) and value.strip() else None
