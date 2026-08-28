from __future__ import annotations

import copy
import gc
import inspect
import pickle
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from importlib import import_module
from weakref import ref

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from starlette.requests import Request

from backend.app.core.config import get_settings
from backend.app.core.demo_auth import (
    USERS,
    DemoUser,
    demo_user_from_serialized,
    get_demo_user,
)
from backend.app.core.session_auth import create_session_token, serialize_auth_user
from backend.app.models import (
    AgentRun,
    AgentWorkflowEvidenceRef,
    AgentWorkflowThread,
    AuditLog,
    AuthUser,
    HistoryEvent,
    ReviewItem,
    ReviewItemEvidenceRef,
    Source,
)
from backend.app.review.actors import (
    CreateNewPromotion,
    ReuseExistingPromotion,
    ReviewResolutionActor,
    auto_review_actor,
    human_review_actor,
)
from backend.app.review.transitions import (
    CanonicalReviewEvidenceStalenessResolver,
    InternalReviewTransitionService,
    ReviewTransitionService,
)
from backend.app.schemas.auto_review import AUTO_REVIEW_POLICY_VERSION
from backend.app.services.audit import record_review_resolution_audit


def _seed_item(
    db: Session,
    *,
    item_type: str = 'history_event',
    permission_level: str = 'internal',
) -> ReviewItem:
    payload = {
        'history_event': {
            'title': 'Actor boundary introduced',
            'reason': 'The approved design requires one locked transition core.',
        },
        'timeline_event': {
            'title': 'Actor boundary introduced',
            'result_summary': 'The review transition now distinguishes resolution actors.',
        },
    }[item_type]
    item = ReviewItem(
        item_type=item_type,
        payload=payload,
        source_links=['https://mail.mock/evidence/actor-boundary'],
        source_snippets=['The source directly supports the bounded test claim.'],
        confidence_score=0.99,
        permission_level=permission_level,
        status='pending_review',
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def _seed_bound_item(db: Session) -> tuple[ReviewItem, Source]:
    thread_id = 'actor-stale-thread'
    signature = 'a' * 64
    source = Source(
        source_type='drive',
        source_id='drive:actor-stale-source',
        source_url='https://drive.mock/actor-stale-source',
        title='Actor stale source',
        permission_level='internal',
        raw_metadata={},
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=signature,
        connector_content_signature='connector-display-only',
    )
    thread = AgentWorkflowThread(
        thread_id=thread_id,
        workflow_name='company_memory_review',
        graph_version='company-memory-review-v2.1-auto-review',
        checkpoint_thread_id='checkpoint:actor-stale-thread',
        checkpoint_store='memory',
        owner_subject_id=USERS['admin'].id,
        security_scope_id='default',
        input_hash='1' * 64,
        evidence_version_hash='2' * 64,
        status='awaiting_review',
    )
    db.add_all([source, thread])
    db.flush()
    run = AgentRun(
        agent_name='history_agent',
        prompt_version='history-extraction:c5-v1',
        status='complete',
        source_window='bounded-test-window',
        cache_key='actor-stale-run',
        model_name='gpt-5.4-mini-2026-03-17',
        generation_provider='openai',
        generation_reasoning_effort='none',
        generation_route_version='auto-review-extraction-route:v1',
        generation_output_contract_version='history-candidate:c5-v1',
        permission_level='internal',
        workflow_thread_id=thread_id,
        effect_key='actor-stale-effect',
    )
    db.add(run)
    db.flush()
    workflow_ref = AgentWorkflowEvidenceRef(
        workflow_thread_id=thread_id,
        ordinal=1,
        canonical_source_type='drive',
        canonical_table='sources',
        canonical_row_id=source.id,
        document_version_id=None,
        external_revision=None,
        content_signature=signature,
        permission_level_snapshot='internal',
        content_fingerprint='3' * 64,
    )
    item = ReviewItem(
        item_type='history_event',
        payload={
            'title': 'Canonical evidence was bound',
            'reason': 'The candidate records the exact source version.',
        },
        source_links=[source.source_url],
        source_snippets=['The candidate records the exact source version.'],
        confidence_score=0.99,
        permission_level='internal',
        status='pending_review',
        workflow_thread_id=thread_id,
        candidate_key='actor-stale-candidate',
        agent_run_id=run.id,
        candidate_contract_version='c5-v1',
    )
    db.add_all([workflow_ref, item])
    db.flush()
    db.add(
        ReviewItemEvidenceRef(
            review_item_id=item.id,
            workflow_thread_id=thread_id,
            workflow_evidence_ref_id=workflow_ref.id,
            candidate_slot_ordinal=1,
            message_content_fingerprint='4' * 64,
            fingerprint_key_version='test-key-v1',
            fingerprint_key_material_verifier='5' * 64,
        )
    )
    db.commit()
    db.refresh(item)
    db.refresh(source)
    return item, source


def test_human_adapter_preserves_exact_permission_levels() -> None:
    actor = human_review_actor(USERS['admin'])

    assert actor.subject_id == USERS['admin'].id
    assert actor.actor_type == 'human'
    assert set(actor.allowed_permission_levels) == USERS['admin'].permission_levels
    assert actor.capabilities == frozenset(
        {'human_review', 'auto_review_rollout_admin'}
    )
    assert actor.policy_version is None


@pytest.mark.parametrize(
    'actor_fields',
    [
        {
            'subject_id': 'forged-human',
            'actor_type': 'human',
            'allowed_permission_levels': ('restricted',),
            'capabilities': frozenset({'human_review'}),
        },
        {
            'subject_id': USERS['admin'].id,
            'actor_type': 'human',
            'allowed_permission_levels': ('public', 'root'),
            'capabilities': frozenset({'human_review'}),
        },
        {
            'subject_id': USERS['admin'].id,
            'actor_type': 'human',
            'allowed_permission_levels': ('public',),
            'capabilities': frozenset({'human_review', 'superuser'}),
        },
        {
            'subject_id': 'system:auto-review',
            'actor_type': 'robot',
            'allowed_permission_levels': ('public', 'internal'),
            'capabilities': frozenset({'auto_review'}),
            'policy_version': AUTO_REVIEW_POLICY_VERSION,
        },
        {
            'subject_id': 'system:auto-review',
            'actor_type': 'human',
            'allowed_permission_levels': ('public', 'internal'),
            'capabilities': frozenset({'human_review'}),
        },
    ],
)
def test_callers_cannot_construct_review_resolution_authority(
    actor_fields: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        ReviewResolutionActor(**actor_fields)


def test_human_adapter_rejects_forged_known_canonical_identity() -> None:
    canonical = USERS['admin']
    forged = DemoUser(
        id=canonical.id,
        email='forged-admin@example.invalid',
        role=canonical.role,
        permission_levels=set(canonical.permission_levels),
        name=canonical.name,
        title=canonical.title,
        department=canonical.department,
    )

    with pytest.raises(ValueError, match='canonical'):
        human_review_actor(forged)


def test_human_adapter_rejects_equal_value_clone_of_canonical_demo_user() -> None:
    canonical = USERS['admin']
    equal_value_clone = DemoUser(
        id=canonical.id,
        email=canonical.email,
        role=canonical.role,
        permission_levels=set(canonical.permission_levels),
        name=canonical.name,
        title=canonical.title,
        department=canonical.department,
        aliases=canonical.aliases,
    )

    assert equal_value_clone == canonical
    with pytest.raises(ValueError, match='canonical'):
        human_review_actor(equal_value_clone)


def test_human_adapter_rejects_unknown_demo_user_identity() -> None:
    unknown = DemoUser(
        id='unknown-reviewer',
        email='unknown-reviewer@example.invalid',
        role='admin',
        permission_levels={'public', 'internal', 'restricted'},
        name='Unknown Reviewer',
        title='Unknown',
        department='Unknown',
    )

    with pytest.raises(ValueError, match='canonical'):
        human_review_actor(unknown)


def test_direct_auth_user_serialization_projection_is_not_authenticated(
    db_session: Session,
) -> None:
    auth_user = AuthUser(
        external_id='direct-serialization-user',
        email='direct-serialization@example.invalid',
        display_name='Direct Serialization User',
        role='reviewer',
        department='Security',
        title='Reviewer',
        status='active',
        permission_levels=['public', 'internal'],
    )
    db_session.add(auth_user)
    db_session.commit()
    projection = demo_user_from_serialized(serialize_auth_user(auth_user))

    with pytest.raises(ValueError):
        human_review_actor(projection)


def test_real_session_projection_requires_exact_unmutated_identity(
    db_session: Session,
) -> None:
    auth_user = AuthUser(
        external_id='session-provenance-reviewer',
        email='session-provenance@example.invalid',
        display_name='Session Provenance Reviewer',
        role='reviewer',
        department='Security',
        title='Reviewer',
        status='active',
        permission_levels=['public', 'internal'],
    )
    db_session.add(auth_user)
    db_session.commit()
    settings = get_settings()
    session_token = create_session_token(auth_user.id, settings)
    request = Request(
        {
            'type': 'http',
            'headers': [
                (
                    b'cookie',
                    (
                        f'{settings.auth_session_cookie_name}={session_token}'
                    ).encode(),
                )
            ],
        }
    )
    projection = get_demo_user(request, db_session, 'viewer')
    clone = DemoUser(
        id=projection.id,
        email=projection.email,
        role=projection.role,
        permission_levels=set(projection.permission_levels),
        name=projection.name,
        title=projection.title,
        department=projection.department,
        aliases=projection.aliases,
    )

    human_review_actor(projection)
    assert clone == projection
    with pytest.raises(ValueError):
        human_review_actor(clone)
    projection.permission_levels.add('restricted')
    with pytest.raises(ValueError):
        human_review_actor(projection)


def test_authenticated_projection_registry_is_weak_and_not_module_exposed(
    db_session: Session,
) -> None:
    auth_user = AuthUser(
        external_id='session-gc-reviewer',
        email='session-gc@example.invalid',
        display_name='Session GC Reviewer',
        role='reviewer',
        department='Security',
        title='Reviewer',
        status='active',
        permission_levels=['public', 'internal'],
    )
    db_session.add(auth_user)
    db_session.commit()
    settings = get_settings()
    session_token = create_session_token(auth_user.id, settings)
    request = Request(
        {
            'type': 'http',
            'headers': [
                (
                    b'cookie',
                    (
                        f'{settings.auth_session_cookie_name}={session_token}'
                    ).encode(),
                )
            ],
        }
    )
    projection = get_demo_user(request, db_session, 'viewer')
    projection_reference = ref(projection)
    demo_auth_module = import_module('backend.app.core.demo_auth')

    for name in (
        '_register_authenticated_demo_user',
        '_authenticated_demo_users',
        '_authentication_registry',
        '_build_authenticated_demo_user_boundary',
    ):
        assert name not in vars(demo_auth_module)

    del projection
    gc.collect()
    assert projection_reference() is None


def test_review_package_root_does_not_export_authority_constructors() -> None:
    review_package = import_module('backend.app.review')

    for name in (
        'ReviewResolutionActor',
        'CreateNewPromotion',
        'ReuseExistingPromotion',
        'human_review_actor',
        'auto_review_actor',
        '_issue_actor',
        '_build_actor_boundary',
        '_assert_review_resolution_actor',
    ):
        assert not hasattr(review_package, name)


def test_actor_module_has_no_unrestricted_issuer_or_registry_surface() -> None:
    actor_module = import_module('backend.app.review.actors')
    module_surface = vars(actor_module)

    for name in (
        '_issue_actor',
        '_is_server_issued_actor',
        '_build_actor_issuer',
        '_build_actor_boundary',
        '_issued_actors',
        '_actor_registry',
    ):
        assert name not in module_surface

    arbitrary_claim_parameters = {
        'subject_id',
        'allowed_permission_levels',
        'capabilities',
    }
    for name, value in module_surface.items():
        if not inspect.isfunction(value) or value.__module__ != actor_module.__name__:
            continue
        parameters = set(inspect.signature(value).parameters)
        assert not arbitrary_claim_parameters.issubset(parameters), name


def test_importable_raw_issuer_cannot_promote_forged_restricted_authority(
    db_session: Session,
) -> None:
    actor_module = import_module('backend.app.review.actors')
    item = _seed_item(db_session, permission_level='restricted')

    with pytest.raises((AttributeError, TypeError, ValueError, HTTPException)):
        forged_actor = actor_module._issue_actor(
            subject_id='forged-subject',
            actor_type='human',
            allowed_permission_levels=('restricted',),
            capabilities=frozenset({'human_review'}),
        )
        ReviewTransitionService().transition(
            db=db_session,
            item_id=item.id,
            action='approve',
            actor=forged_actor,
        )

    db_session.refresh(item)
    assert item.status == 'pending_review'
    assert item.reviewer_id is None
    assert db_session.scalar(select(func.count()).select_from(HistoryEvent)) == 0


def test_constrained_actor_factories_issue_validator_accepted_identities() -> None:
    actor_module = import_module('backend.app.review.actors')

    actor_module._assert_review_resolution_actor(
        human_review_actor(USERS['admin'])
    )
    actor_module._assert_review_resolution_actor(
        auto_review_actor(policy_version=AUTO_REVIEW_POLICY_VERSION)
    )


def test_actor_identity_registry_does_not_retain_collected_actors() -> None:
    actor = human_review_actor(USERS['admin'])
    actor_reference = ref(actor)

    del actor
    gc.collect()

    assert actor_reference() is None


def test_direct_forged_human_cannot_approve_restricted_item(
    db_session: Session,
) -> None:
    item = _seed_item(db_session, permission_level='restricted')

    with pytest.raises((TypeError, ValueError, HTTPException)):
        actor = ReviewResolutionActor(
            subject_id='forged-subject',
            actor_type='human',
            allowed_permission_levels=('restricted',),
            capabilities=frozenset({'human_review'}),
        )
        ReviewTransitionService().transition(
            db=db_session,
            item_id=item.id,
            action='approve',
            actor=actor,
        )

    db_session.refresh(item)
    assert item.status == 'pending_review'
    assert item.reviewer_id is None


def test_canonical_actor_bearer_material_cannot_mint_restricted_authority(
    db_session: Session,
) -> None:
    canonical_employee = human_review_actor(USERS['hanvv-employee'])
    copied_authority = getattr(canonical_employee, '_authority', None)
    item = _seed_item(db_session, permission_level='restricted')

    with pytest.raises((TypeError, ValueError, HTTPException)):
        forged_actor = ReviewResolutionActor(
            subject_id='forged-subject',
            actor_type='human',
            allowed_permission_levels=('restricted',),
            capabilities=frozenset({'human_review'}),
            _authority=copied_authority,  # type: ignore[call-arg]
        )
        ReviewTransitionService().transition(
            db=db_session,
            item_id=item.id,
            action='approve',
            actor=forged_actor,
        )

    db_session.refresh(item)
    assert item.status == 'pending_review'
    assert item.reviewer_id is None
    assert db_session.scalar(select(func.count()).select_from(HistoryEvent)) == 0


def test_issued_actor_has_no_bearer_material_and_is_immutable_slotted() -> None:
    actor = human_review_actor(USERS['admin'])

    assert not hasattr(actor, '_authority')
    assert not hasattr(actor, '__dict__')
    with pytest.raises((AttributeError, FrozenInstanceError)):
        actor.subject_id = 'forged-subject'  # type: ignore[misc]


def test_object_new_same_value_clone_is_not_server_issued(
    db_session: Session,
) -> None:
    canonical = human_review_actor(USERS['admin'])
    clone = object.__new__(ReviewResolutionActor)
    for field_name in (
        'subject_id',
        'actor_type',
        'allowed_permission_levels',
        'capabilities',
        'policy_version',
    ):
        object.__setattr__(clone, field_name, getattr(canonical, field_name))
    if hasattr(canonical, '_authority'):
        object.__setattr__(clone, '_authority', canonical._authority)
    item = _seed_item(db_session, permission_level='restricted')

    with pytest.raises((TypeError, ValueError, HTTPException)):
        ReviewTransitionService().transition(
            db=db_session,
            item_id=item.id,
            action='approve',
            actor=clone,
        )

    db_session.refresh(item)
    assert item.status == 'pending_review'
    assert db_session.scalar(select(func.count()).select_from(HistoryEvent)) == 0


def test_server_issued_actor_rejects_object_setattr_claim_mutation(
    db_session: Session,
) -> None:
    actor = human_review_actor(USERS['hanvv-employee'])
    object.__setattr__(actor, 'subject_id', 'forged-subject')
    object.__setattr__(actor, 'allowed_permission_levels', ('restricted',))
    object.__setattr__(actor, 'capabilities', frozenset({'human_review'}))
    item = _seed_item(db_session, permission_level='restricted')

    with pytest.raises((TypeError, ValueError, HTTPException)):
        ReviewTransitionService().transition(
            db=db_session,
            item_id=item.id,
            action='approve',
            actor=actor,
        )

    db_session.refresh(item)
    assert item.status == 'pending_review'
    assert db_session.scalar(select(func.count()).select_from(HistoryEvent)) == 0


def test_actor_copy_deepcopy_replace_and_pickle_are_rejected() -> None:
    actor = auto_review_actor(policy_version=AUTO_REVIEW_POLICY_VERSION)

    with pytest.raises(TypeError):
        copy.copy(actor)
    with pytest.raises(TypeError):
        copy.deepcopy(actor)
    with pytest.raises(TypeError):
        replace(actor)
    with pytest.raises(TypeError):
        pickle.dumps(actor)


def test_auto_actor_has_only_public_internal_and_auto_review() -> None:
    actor = auto_review_actor(policy_version=AUTO_REVIEW_POLICY_VERSION)

    assert actor.subject_id == 'system:auto-review'
    assert actor.actor_type == 'auto_policy'
    assert actor.allowed_permission_levels == ('public', 'internal')
    assert actor.capabilities == frozenset({'auto_review'})
    assert actor.policy_version == AUTO_REVIEW_POLICY_VERSION


def test_approval_directives_reject_kind_and_type_contradictions() -> None:
    with pytest.raises(ValueError):
        CreateNewPromotion(kind='reuse_existing')  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ReuseExistingPromotion(
            expected_type='decision_record',  # type: ignore[arg-type]
            expected_id=1,
            expected_claim_fingerprint='a' * 64,
            expected_companion_id=None,
            expected_companion_claim_fingerprint=None,
        )
    with pytest.raises(ValueError):
        ReuseExistingPromotion(
            expected_type='history_event',
            expected_id=1,
            expected_claim_fingerprint='a' * 64,
            expected_companion_id=None,
            expected_companion_claim_fingerprint=None,
            kind='create_new',  # type: ignore[arg-type]
        )


def test_transition_rejects_unknown_approval_directive_runtime_type(
    db_session: Session,
) -> None:
    item = _seed_item(db_session)

    with pytest.raises((TypeError, ValueError)):
        ReviewTransitionService().transition(
            db=db_session,
            item_id=item.id,
            action='approve',
            actor=human_review_actor(USERS['admin']),
            approval_directive=object(),  # type: ignore[arg-type]
        )

    db_session.refresh(item)
    assert item.status == 'pending_review'


def test_public_request_cannot_supply_actor_type_capability_or_directive(
    client,
    db_session: Session,
) -> None:
    item = _seed_item(db_session)

    response = client.post(
        f'/api/v1/review/{item.id}/approve',
        params={
            'actor_type': 'auto_policy',
            'capability': 'auto_review',
            'approval_directive': 'reuse_existing',
        },
        headers={
            'X-Demo-User': 'viewer',
            'X-Review-Actor-Type': 'auto_policy',
            'X-Review-Capability': 'auto_review',
        },
        json={
            'actor_type': 'auto_policy',
            'capability': 'auto_review',
            'approval_directive': {'kind': 'reuse_existing'},
        },
    )

    assert response.status_code == 200
    db_session.refresh(item)
    assert item.reviewer_id == USERS['viewer'].id
    assert item.resolution_source == 'human'
    assert item.resolution_policy_version is None
    assert item.auto_validation_id is None


def test_real_local_cookie_session_preserves_all_public_review_actions(
    client,
    db_session: Session,
) -> None:
    login = client.post(
        '/api/v1/auth/login',
        json={'email': USERS['admin'].email},
    )
    assert login.status_code == 200
    approve_item = _seed_item(db_session)
    reject_item = _seed_item(db_session, item_type='timeline_event')
    evidence_item = _seed_item(db_session)

    approved = client.post(f'/api/v1/review/{approve_item.id}/approve')
    replayed = client.post(f'/api/v1/review/{approve_item.id}/approve')
    rejected = client.post(f'/api/v1/review/{reject_item.id}/reject')
    needs_more = client.post(
        f'/api/v1/review/{evidence_item.id}/request-more-evidence',
        json={'note': 'Attach the exact document version.'},
    )

    assert approved.status_code == 200
    assert approved.json()['status'] == 'approved'
    assert approved.json()['replayed'] is False
    assert replayed.status_code == 200
    assert replayed.json()['status'] == 'approved'
    assert replayed.json()['replayed'] is True
    assert rejected.status_code == 200
    assert rejected.json()['status'] == 'rejected'
    assert needs_more.status_code == 200
    assert needs_more.json()['status'] == 'needs_more_evidence'
    assert needs_more.json()['payload']['needs_more_evidence']['note'] == (
        'Attach the exact document version.'
    )


def test_real_cookie_session_accepts_persisted_google_identity_shape(
    client,
    db_session: Session,
) -> None:
    auth_user = AuthUser(
        external_id='google-oauth2:reviewer-42',
        email='google-reviewer@example.invalid',
        display_name='Google Reviewer',
        role='reviewer',
        department='Product',
        title='Product Reviewer',
        status='active',
        permission_levels=['public', 'internal'],
    )
    db_session.add(auth_user)
    db_session.commit()
    settings = get_settings()
    client.cookies.set(
        settings.auth_session_cookie_name,
        create_session_token(auth_user.id, settings),
        domain='testserver.local',
        path='/',
    )
    item = _seed_item(db_session)

    response = client.post(f'/api/v1/review/{item.id}/approve')

    assert response.status_code == 200
    assert response.json()['status'] == 'approved'
    assert response.json()['reviewer_id'] == auth_user.external_id


def test_auto_actor_cannot_reject_request_evidence_or_bulk_review(
    db_session: Session,
) -> None:
    actor = auto_review_actor(policy_version=AUTO_REVIEW_POLICY_VERSION)
    service = ReviewTransitionService()
    reject_item = _seed_item(db_session)
    evidence_item = _seed_item(db_session, item_type='timeline_event')

    for item, action in (
        (reject_item, 'reject'),
        (evidence_item, 'needs_more_evidence'),
    ):
        with pytest.raises(HTTPException) as exc_info:
            service.transition(
                db=db_session,
                item_id=item.id,
                action=action,
                actor=actor,
            )
        assert exc_info.value.status_code == 403

    with pytest.raises(HTTPException) as exc_info:
        service.transition_many(
            db=db_session,
            item_ids=[reject_item.id, evidence_item.id],
            action='approve',
            actor=actor,
        )
    assert exc_info.value.status_code == 403
    assert reject_item.status == 'pending_review'
    assert evidence_item.status == 'pending_review'


def test_only_internal_stale_directive_with_locked_canonical_drift_marks_needs_more(
    db_session: Session,
) -> None:
    item, source = _seed_bound_item(db_session)
    service = InternalReviewTransitionService(
        evidence_staleness_resolver=CanonicalReviewEvidenceStalenessResolver()
    )
    source.server_content_signature = 'b' * 64
    db_session.commit()

    result = service.mark_evidence_stale(db=db_session, item_id=item.id)

    db_session.refresh(item)
    assert result.status == 'needs_more_evidence'
    assert result.promotion is None
    assert item.reviewer_id == 'system:auto-review'
    assert item.payload['needs_more_evidence'] == {
        'requested_at': item.payload['needs_more_evidence']['requested_at'],
        'requested_by': 'system:auto-review',
        'reason_code': 'evidence_version_changed',
        'previous_status': 'pending_review',
    }
    assert item.payload['needs_more_evidence']['requested_at'].endswith('+00:00')


def test_public_or_current_evidence_cannot_invoke_stale_directive(
    db_session: Session,
) -> None:
    item, _source = _seed_bound_item(db_session)
    internal = InternalReviewTransitionService(
        evidence_staleness_resolver=CanonicalReviewEvidenceStalenessResolver()
    )
    auto_actor = auto_review_actor(policy_version=AUTO_REVIEW_POLICY_VERSION)

    with pytest.raises(ValueError, match='canonical evidence is current'):
        internal.mark_evidence_stale(db=db_session, item_id=item.id)
    with pytest.raises(HTTPException) as exc_info:
        ReviewTransitionService().transition(
            db=db_session,
            item_id=item.id,
            action='needs_more_evidence',
            actor=auto_actor,
        )
    assert exc_info.value.status_code == 403
    assert item.status == 'pending_review'


def test_wrong_policy_auto_actor_cannot_approve(db_session: Session) -> None:
    item = _seed_item(db_session)
    wrong_policy_actor = auto_review_actor(policy_version='auto-review-policy:wrong')

    with pytest.raises(HTTPException) as exc_info:
        ReviewTransitionService().transition(
            db=db_session,
            item_id=item.id,
            action='approve',
            actor=wrong_policy_actor,
            approval_directive=CreateNewPromotion(),
        )

    assert exc_info.value.status_code == 403
    assert item.status == 'pending_review'


def test_existing_human_approve_reject_needs_more_and_replay_are_unchanged(
    db_session: Session,
) -> None:
    actor = human_review_actor(USERS['admin'])
    approve_item = _seed_item(db_session)
    reject_item = _seed_item(db_session, item_type='timeline_event')
    evidence_item = _seed_item(db_session)
    service = ReviewTransitionService()

    approved = service.transition(
        db=db_session,
        item_id=approve_item.id,
        action='approve',
        actor=actor,
    )
    replay = service.transition(
        db=db_session,
        item_id=approve_item.id,
        action='approve',
        actor=actor,
    )
    rejected = service.transition(
        db=db_session,
        item_id=reject_item.id,
        action='reject',
        actor=actor,
    )
    needs_more = service.transition(
        db=db_session,
        item_id=evidence_item.id,
        action='needs_more_evidence',
        actor=actor,
        note='Add the exact source version.',
    )

    assert approved.status == 'approved'
    assert replay.replayed is True
    assert replay.promotion == approved.promotion
    assert rejected.status == 'rejected'
    assert needs_more.status == 'needs_more_evidence'
    assert evidence_item.payload['needs_more_evidence']['note'] == (
        'Add the exact source version.'
    )
    for item in (approve_item, reject_item, evidence_item):
        assert item.resolution_source == 'human'
        assert item.resolution_policy_version is None
        assert item.auto_validation_id is None


def test_audit_writer_does_not_log_raw_reason_or_source_content(
    db_session: Session,
) -> None:
    actor = human_review_actor(USERS['admin'])

    record_review_resolution_audit(
        db=db_session,
        actor=actor,
        human_user=USERS['admin'],
        action='review.approve',
        review_item_id=41,
        outcome='approved',
        metadata={
            'item_type': 'history_event',
            'effect_count': 1,
            'reason': 'raw model rationale must not be stored',
            'source_content': 'raw source content must not be stored',
            'source_url': 'https://private.invalid/raw',
            'estimated_cost_usd': Decimal('0.001234'),
        },
    )
    db_session.commit()

    audit = db_session.scalars(select(AuditLog)).one()
    serialized = repr(audit.metadata_)
    assert 'raw model rationale' not in serialized
    assert 'raw source content' not in serialized
    assert 'private.invalid' not in serialized
    assert audit.metadata_ == {
        'actor_type': 'human',
        'review_item_id': 41,
        'outcome': 'approved',
        'item_type': 'history_event',
        'effect_count': 1,
        'estimated_cost_usd': '0.001234',
    }


def test_audit_writer_rejects_raw_content_laundered_through_typed_fields(
    db_session: Session,
) -> None:
    actor = human_review_actor(USERS['admin'])

    record_review_resolution_audit(
        db=db_session,
        actor=actor,
        human_user=USERS['admin'],
        action='review.approve',
        review_item_id=42,
        outcome='approved',
        metadata={
            'item_type': 'raw model rationale: customer secret',
            'policy_version': 'raw source evidence: private',
            'permission_level': 'private customer transcript',
            'validator_prompt_version': 'raw prompt: customer secret',
        },
    )
    db_session.commit()

    audit = db_session.scalars(select(AuditLog)).one()
    serialized = repr(audit.metadata_)
    assert 'customer secret' not in serialized
    assert 'source evidence' not in serialized
    assert 'customer transcript' not in serialized
    assert 'raw prompt' not in serialized
    assert audit.metadata_ == {
        'actor_type': 'human',
        'review_item_id': 42,
        'outcome': 'approved',
    }


def test_review_audit_rejects_forged_human_projection_and_raw_target_id(
    db_session: Session,
) -> None:
    actor = human_review_actor(USERS['admin'])
    canonical = USERS['admin']
    forged_projection = DemoUser(
        id=canonical.id,
        email='forged-audit@example.invalid',
        role=canonical.role,
        permission_levels=set(canonical.permission_levels),
        name=canonical.name,
        title=canonical.title,
        department=canonical.department,
    )

    with pytest.raises(ValueError, match='canonical'):
        record_review_resolution_audit(
            db=db_session,
            actor=actor,
            human_user=forged_projection,
            action='review.approve',
            review_item_id=43,
            outcome='approved',
        )
    with pytest.raises(ValueError, match='target'):
        record_review_resolution_audit(
            db=db_session,
            actor=actor,
            human_user=canonical,
            action='review.approve',
            review_item_id=43,
            outcome='approved',
            target_type='review_workflow',
            target_id='raw source evidence: private',
        )

    db_session.flush()
    assert db_session.scalars(select(AuditLog)).all() == []


def test_system_resolution_audit_uses_fixed_schema_projection_without_fake_user(
    db_session: Session,
) -> None:
    actor = auto_review_actor(policy_version=AUTO_REVIEW_POLICY_VERSION)

    record_review_resolution_audit(
        db=db_session,
        actor=actor,
        action='review.auto_approve',
        review_item_id=73,
        outcome='approved',
        metadata={
            'policy_version': AUTO_REVIEW_POLICY_VERSION,
            'validation_id': 9,
            'effect_count': 2,
        },
    )
    db_session.commit()

    audit = db_session.scalars(select(AuditLog)).one()
    assert audit.actor_id == 'system:auto-review'
    assert audit.actor_email == 'system:auto-review@paraworks.invalid'
    assert audit.actor_role == 'system'
    assert audit.target_type == 'review_item'
    assert audit.target_id == '73'
    assert audit.metadata_ == {
        'actor_type': 'auto_policy',
        'review_item_id': 73,
        'outcome': 'approved',
        'policy_version': AUTO_REVIEW_POLICY_VERSION,
        'validation_id': 9,
        'effect_count': 2,
    }
