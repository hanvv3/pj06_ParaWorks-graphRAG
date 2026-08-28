from collections.abc import Mapping
from decimal import Decimal

from sqlalchemy.orm import Session

from backend.app.core.demo_auth import DemoUser
from backend.app.core.redaction import redact_secret_text
from backend.app.models import AuditLog
from backend.app.review.actors import (
    SYSTEM_AUTO_REVIEW_ACTOR_EMAIL,
    SYSTEM_AUTO_REVIEW_ACTOR_ID,
    SYSTEM_AUTO_REVIEW_ACTOR_ROLE,
    ReviewResolutionActor,
)

_REVIEW_RESOLUTION_METADATA_KEYS = frozenset({
    'effect_count',
    'estimated_cost_usd',
    'failed_count',
    'input_tokens',
    'item_type',
    'note_present',
    'output_tokens',
    'permission_level',
    'policy_version',
    'rejected_count',
    'replayed',
    'replayed_count',
    'skipped_count',
    'source_count',
    'validation_id',
    'validator_output_contract_version',
    'validator_prompt_version',
})
_REVIEW_RESOLUTION_COUNT_KEYS = frozenset({
    'effect_count',
    'failed_count',
    'input_tokens',
    'output_tokens',
    'rejected_count',
    'replayed_count',
    'skipped_count',
    'source_count',
    'validation_id',
})
_MAX_REVIEW_AUDIT_COUNT = 1_000_000_000


def record_audit_log(
    *,
    db: Session,
    actor: DemoUser,
    action: str,
    target_type: str,
    target_id: str | int | None = None,
    status: str = 'success',
    metadata: Mapping[str, object] | None = None,
) -> AuditLog:
    audit = AuditLog(
        actor_id=actor.id,
        actor_email=actor.email,
        actor_role=actor.role,
        action=action,
        target_type=target_type,
        target_id=str(target_id) if target_id is not None else None,
        status=status,
        metadata_=_sanitize_metadata(metadata or {}),
    )
    db.add(audit)
    return audit


def record_review_resolution_audit(
    *,
    db: Session,
    actor: ReviewResolutionActor,
    action: str,
    review_item_id: int,
    outcome: str,
    metadata: Mapping[str, object] | None = None,
    human_user: DemoUser | None = None,
    target_type: str = 'review_item',
    target_id: str | int | None = None,
    status: str = 'success',
) -> AuditLog:
    if actor.actor_type == 'human':
        if human_user is None or human_user.id != actor.subject_id:
            raise ValueError('Human review audit identity does not match actor')
        actor_id = human_user.id
        actor_email = human_user.email
        actor_role = human_user.role
    else:
        if human_user is not None:
            raise ValueError('System review audit cannot use a human user projection')
        actor_id = SYSTEM_AUTO_REVIEW_ACTOR_ID
        actor_email = SYSTEM_AUTO_REVIEW_ACTOR_EMAIL
        actor_role = SYSTEM_AUTO_REVIEW_ACTOR_ROLE
    bounded_metadata: dict[str, object] = {
        'actor_type': actor.actor_type,
        'review_item_id': review_item_id,
        'outcome': _bounded_review_audit_text(outcome),
    }
    for key, value in (metadata or {}).items():
        if key not in _REVIEW_RESOLUTION_METADATA_KEYS:
            continue
        if key in _REVIEW_RESOLUTION_COUNT_KEYS:
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            if value < 0 or value > _MAX_REVIEW_AUDIT_COUNT:
                continue
            bounded_metadata[key] = value
        elif key == 'estimated_cost_usd':
            if isinstance(value, Decimal) and value.is_finite() and value >= 0:
                bounded_metadata[key] = format(value, 'f')
        elif isinstance(value, bool):
            bounded_metadata[key] = value
        elif isinstance(value, str):
            bounded_metadata[key] = _bounded_review_audit_text(value)
    audit = AuditLog(
        actor_id=actor_id,
        actor_email=actor_email,
        actor_role=actor_role,
        action=_bounded_review_audit_text(action),
        target_type=_bounded_review_audit_text(target_type),
        target_id=str(target_id if target_id is not None else review_item_id),
        status=_bounded_review_audit_text(status),
        metadata_=bounded_metadata,
    )
    db.add(audit)
    return audit


def serialize_audit_log(log: AuditLog) -> dict[str, object]:
    return {
        'id': log.id,
        'actor_id': log.actor_id,
        'actor_email': log.actor_email,
        'actor_role': log.actor_role,
        'action': log.action,
        'target_type': log.target_type,
        'target_id': log.target_id,
        'status': log.status,
        'metadata': log.metadata_ or {},
        'created_at': log.created_at.isoformat(),
    }


def _sanitize_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    sanitized: dict[str, object] = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            sanitized[key] = redact_secret_text(value)
        elif isinstance(value, list):
            sanitized[key] = [redact_secret_text(item) if isinstance(item, str) else item for item in value]
        else:
            sanitized[key] = value
    return sanitized


def _bounded_review_audit_text(value: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > 128:
        raise ValueError('Review audit text is outside the bounded schema')
    return normalized
