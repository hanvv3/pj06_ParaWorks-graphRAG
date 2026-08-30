from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from backend.app.api.v1.review import (
    _auto_review_audit,
    _auto_review_summary,
    router,
)
from backend.app.schemas.review import (
    AutoReviewAuditRequest,
    RevokeAutoApprovalRequest,
)


class _ProjectionSession:
    def __init__(self, *, validation=None, scalar_rows=()) -> None:
        self.validation = validation
        self.scalar_rows = iter(scalar_rows)

    def get(self, _model, _identity):
        return self.validation

    def scalar(self, _statement):
        return next(self.scalar_rows)


def _auto_item():
    return SimpleNamespace(
        id=17,
        workflow_thread_id='workflow-17',
        resolution_source='auto_policy',
        auto_validation_id=29,
    )


def _canonical_validation(**changes):
    values = {
        'review_item_id': 17,
        'workflow_thread_id': 'workflow-17',
        'status': 'completed',
        'validator_model': 'gpt-5.6-terra',
        'reasoning_effort': 'medium',
        'validator_prompt_version': 'auto-review-validation:v2',
        'validator_output_contract_version': 'candidate-validation-batch:v1',
        'policy_version': 'auto-review-policy:v1',
        'minimum_entailment_score': Decimal('0.9900'),
        'claim_results': [
            {'field_key': 'title', 'verdict': 'supported'},
            {'field_key': 'summary', 'verdict': 'supported'},
        ],
        'policy_reason_codes': ['direct_fact_supported'],
        'completed_at': datetime(2026, 8, 30, tzinfo=UTC),
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_revoke_request_is_enum_only_and_forbids_free_text_or_extra_fields() -> None:
    request = RevokeAutoApprovalRequest(reason_code='business_withdrawal')
    assert request.model_dump() == {'reason_code': 'business_withdrawal'}

    with pytest.raises(ValidationError):
        RevokeAutoApprovalRequest.model_validate({
            'reason_code': 'business_withdrawal',
            'reason': 'free text must not enter this route',
        })
    with pytest.raises(ValidationError):
        RevokeAutoApprovalRequest(reason_code='unknown')


def test_audit_request_normalizes_bounded_human_reason() -> None:
    request = AutoReviewAuditRequest(
        outcome='confirmed', reason='  reviewed\n against   evidence  '
    )
    assert request.reason == 'reviewed against evidence'
    with pytest.raises(ValidationError):
        AutoReviewAuditRequest(outcome='confirmed', reason='   ')


def test_auto_summary_is_bounded_and_fail_closed_for_internal_codes() -> None:
    item = _auto_item()
    summary = _auto_review_summary(
        _ProjectionSession(validation=_canonical_validation()), item
    )

    assert summary == {
        'validator_model': 'gpt-5.6-terra',
        'reasoning_effort': 'medium',
        'validator_prompt_version': 'auto-review-validation:v2',
        'validator_output_contract_version': 'candidate-validation-batch:v1',
        'policy_version': 'auto-review-policy:v1',
        'supported_substantive_field_count': 2,
        'minimum_entailment_score': 0.99,
        'policy_reason_codes': ['direct_fact_supported'],
        'validated_at': datetime(2026, 8, 30, tzinfo=UTC),
    }
    assert _auto_review_summary(
        _ProjectionSession(
            validation=_canonical_validation(
                policy_reason_codes=['hidden_collision']
            )
        ),
        item,
    ) is None
    assert 'review_item_id' not in summary
    assert 'claim_results' not in summary


def test_corrected_audit_projects_effective_critical_state_without_ids() -> None:
    audit = SimpleNamespace(status='completed', outcome='confirmed')
    correction = SimpleNamespace(
        status='remediation_required',
        effective_outcome='permission_violation',
    )

    projection = _auto_review_audit(
        _ProjectionSession(scalar_rows=(audit, correction)), _auto_item()
    )

    assert projection == {
        'status': 'remediation_required',
        'outcome': 'permission_violation',
        'action_required': True,
    }
    assert 'id' not in projection
    assert 'reason' not in projection


def test_auto_review_routes_publish_only_strict_response_models() -> None:
    routes = {route.path: route for route in router.routes}
    revoke = routes['/review/{item_id}/revoke-auto-approval']
    audit = routes['/review/{item_id}/auto-review-audit']

    assert set(revoke.response_model.model_fields) == {
        'review_item_id',
        'status',
        'replayed',
        'knowledge_remains_trusted',
        'revoked_document_count',
    }
    assert set(audit.response_model.model_fields) == {
        'audit_status',
        'breaker_open',
        'revoke_status',
    }
