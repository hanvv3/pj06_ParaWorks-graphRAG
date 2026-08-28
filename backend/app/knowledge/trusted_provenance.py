from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.core.config import Settings
from backend.app.knowledge.claim_fingerprints import (
    normalized_claim_fingerprint,
    promoted_effect_fingerprint,
)
from backend.app.knowledge.promotion import build_promotion_preview
from backend.app.models import (
    AgentWorkflowEvidenceRef,
    AgentWorkflowThread,
    DecisionRecord,
    HistoryEvent,
    ReviewItem,
    ReviewItemEvidenceRef,
    TimelineEvent,
    Todo,
    TrustedKnowledgeApprovalLink,
    TrustedKnowledgeEvidenceLink,
)

if TYPE_CHECKING:
    from backend.app.review.actors import ReuseExistingPromotion


class TrustedProvenanceMismatch(ValueError):  # noqa: N818 - bounded public sentinel
    """The requested canonical effect bundle is not exact and reusable."""


@dataclass(frozen=True)
class TrustedPromotionEffect:
    kind: str
    knowledge_type: str
    knowledge_id: int
    claim_fingerprint: str


@dataclass(frozen=True)
class TrustedPromotionBundle:
    target_type: str
    record_ids: tuple[int, ...]
    timeline_ids: tuple[int, ...]
    effects: tuple[TrustedPromotionEffect, ...]


def effects_for_created_promotion(
    db: Session,
    *,
    item: ReviewItem,
    record_ids: tuple[int, ...],
    timeline_ids: tuple[int, ...],
    settings: Settings,
) -> TrustedPromotionBundle:
    scope = _security_scope(db, item)
    effects: list[TrustedPromotionEffect] = []
    if item.item_type != 'timeline_event':
        if len(record_ids) != 1:
            raise TrustedProvenanceMismatch('primary promotion effect is incomplete')
        effects.append(
            _effect_from_target(
                db,
                kind='primary',
                knowledge_type=item.item_type,
                knowledge_id=record_ids[0],
                security_scope_id=scope,
                settings=settings,
            )
        )
    if len(timeline_ids) != 1:
        raise TrustedProvenanceMismatch('timeline promotion effect is incomplete')
    effects.append(
        _effect_from_target(
            db,
            kind='primary' if item.item_type == 'timeline_event' else 'companion',
            knowledge_type='timeline_event',
            knowledge_id=timeline_ids[0],
            security_scope_id=scope,
            settings=settings,
        )
    )
    return TrustedPromotionBundle(
        target_type=item.item_type,
        record_ids=record_ids,
        timeline_ids=timeline_ids,
        effects=tuple(effects),
    )


def record_or_verify_explicit_provenance(
    db: Session,
    *,
    item: ReviewItem,
    bundle: TrustedPromotionBundle,
    settings: Settings,
) -> None:
    if item.candidate_contract_version != 'c5-v1':
        return
    if item.workflow_thread_id is None:
        raise TrustedProvenanceMismatch('C.5 promotion requires a workflow identity')
    bindings = tuple(
        db.scalars(
            select(ReviewItemEvidenceRef)
            .where(ReviewItemEvidenceRef.review_item_id == item.id)
            .order_by(ReviewItemEvidenceRef.candidate_slot_ordinal)
        ).all()
    )
    if not bindings:
        raise TrustedProvenanceMismatch('C.5 promotion requires explicit evidence')
    refs = {
        ref.id: ref
        for ref in db.scalars(
            select(AgentWorkflowEvidenceRef).where(
                AgentWorkflowEvidenceRef.id.in_(
                    [binding.workflow_evidence_ref_id for binding in bindings]
                )
            )
        ).all()
    }
    if len(refs) != len(bindings):
        raise TrustedProvenanceMismatch('C.5 promotion evidence is incomplete')
    secret, key_version = fingerprint_secret_bytes(settings)
    verifier = fingerprint_key_material_verifier(secret.decode('utf-8'))
    scope = _security_scope(db, item)
    for effect in bundle.effects:
        existing = db.scalar(
            select(TrustedKnowledgeApprovalLink).where(
                TrustedKnowledgeApprovalLink.review_item_id == item.id,
                TrustedKnowledgeApprovalLink.promotion_effect_kind == effect.kind,
            )
        )
        expected_parent = (
            effect.knowledge_type,
            effect.knowledge_id,
            scope,
            item.resolution_source or 'human',
            effect.claim_fingerprint,
            item.permission_level,
            key_version,
            verifier,
        )
        parent_created = existing is None
        if existing is None:
            existing = TrustedKnowledgeApprovalLink(
                knowledge_type=effect.knowledge_type,
                knowledge_id=effect.knowledge_id,
                review_item_id=item.id,
                security_scope_id=scope,
                promotion_effect_kind=effect.kind,
                resolution_source=item.resolution_source or 'human',
                claim_fingerprint=effect.claim_fingerprint,
                permission_level=item.permission_level,
                fingerprint_key_version=key_version,
                fingerprint_key_material_verifier=verifier,
                active=True,
            )
            db.add(existing)
            db.flush([existing])
        elif _parent_identity(existing) != expected_parent or not existing.active:
            raise TrustedProvenanceMismatch('approval provenance replay mismatch')

        expected_children = {
            _evidence_identity(
                ref=refs[binding.workflow_evidence_ref_id],
                binding=binding,
                secret=secret,
                key_version=key_version,
                verifier=verifier,
            )
            for binding in bindings
        }
        if len(expected_children) != len(bindings):
            raise TrustedProvenanceMismatch('evidence provenance identity collapsed')
        children = tuple(
            db.scalars(
                select(TrustedKnowledgeEvidenceLink).where(
                    TrustedKnowledgeEvidenceLink.approval_link_id == existing.id
                )
            ).all()
        )
        actual_children = {_child_identity(child) for child in children}
        if children and actual_children != expected_children:
            raise TrustedProvenanceMismatch('evidence provenance replay mismatch')
        if not children and not parent_created:
            raise TrustedProvenanceMismatch('evidence provenance replay is incomplete')
        if not children:
            db.add_all(
                [
                    TrustedKnowledgeEvidenceLink(
                        approval_link_id=existing.id,
                        canonical_source_kind=identity[0],
                        canonical_source_id=identity[1],
                        canonical_version_or_signature=identity[2],
                        evidence_hash=identity[3],
                        fingerprint_key_version=identity[4],
                        fingerprint_key_material_verifier=identity[5],
                    )
                    for identity in sorted(expected_children)
                ]
            )
            db.flush()


def find_explicit_promotion(
    db: Session,
    *,
    item: ReviewItem,
    settings: Settings | None = None,
) -> TrustedPromotionBundle | None:
    statement = (
        select(TrustedKnowledgeApprovalLink)
        .where(
            TrustedKnowledgeApprovalLink.review_item_id == item.id,
            TrustedKnowledgeApprovalLink.active.is_(True),
        )
        .order_by(TrustedKnowledgeApprovalLink.promotion_effect_kind.desc())
    )
    if db.get_bind().dialect.name == 'postgresql':
        statement = statement.with_for_update()
    links = tuple(db.scalars(statement).all())
    if not links:
        return None
    effects = tuple(
        TrustedPromotionEffect(
            kind=link.promotion_effect_kind,
            knowledge_type=link.knowledge_type,
            knowledge_id=link.knowledge_id,
            claim_fingerprint=link.claim_fingerprint,
        )
        for link in links
    )
    primary = [effect for effect in effects if effect.kind == 'primary']
    companion = [effect for effect in effects if effect.kind == 'companion']
    if len(primary) != 1 or len(companion) != (0 if item.item_type == 'timeline_event' else 1):
        raise TrustedProvenanceMismatch('explicit promotion bundle is incomplete')
    bundle = TrustedPromotionBundle(
        target_type=item.item_type,
        record_ids=() if item.item_type == 'timeline_event' else (primary[0].knowledge_id,),
        timeline_ids=(
            (primary[0].knowledge_id,)
            if item.item_type == 'timeline_event'
            else (companion[0].knowledge_id,)
        ),
        effects=effects,
    )
    if settings is not None:
        scope = _security_scope(db, item)
        persisted_effects = tuple(
            _effect_from_target(
                db,
                kind=effect.kind,
                knowledge_type=effect.knowledge_type,
                knowledge_id=effect.knowledge_id,
                security_scope_id=scope,
                settings=settings,
                lock=True,
            )
            for effect in effects
        )
        secret, key_version = fingerprint_secret_bytes(settings)
        verifier = fingerprint_key_material_verifier(secret.decode('utf-8'))
        current_identity = all(
            link.fingerprint_key_version == key_version
            and link.fingerprint_key_material_verifier == verifier
            for link in links
        )
        if current_identity:
            if persisted_effects != effects:
                raise TrustedProvenanceMismatch('explicit promotion target has drifted')
            record_or_verify_explicit_provenance(
                db,
                item=item,
                bundle=bundle,
                settings=settings,
            )
        else:
            for link in links:
                _verify_link_children(db, link=link, settings=settings)
    return bundle


def validate_reuse_bundle(
    db: Session,
    *,
    item: ReviewItem,
    directive: ReuseExistingPromotion,
    settings: Settings,
) -> TrustedPromotionBundle:
    if item.item_type != directive.expected_type:
        raise TrustedProvenanceMismatch('candidate type does not match canonical target')
    scope = _security_scope(db, item)
    expected: list[tuple[str, str, int, str]] = [
        ('primary', directive.expected_type, directive.expected_id, directive.expected_claim_fingerprint)
    ]
    if directive.expected_type == 'history_event':
        if directive.expected_companion_id is None or directive.expected_companion_claim_fingerprint is None:
            raise TrustedProvenanceMismatch('history reaffirmation requires one companion')
        expected.append(
            (
                'companion',
                'timeline_event',
                directive.expected_companion_id,
                directive.expected_companion_claim_fingerprint,
            )
        )
    elif directive.expected_companion_id is not None:
        raise TrustedProvenanceMismatch('timeline reaffirmation has no companion')

    effects: list[TrustedPromotionEffect] = []
    link_review_sets: list[set[int]] = []
    secret, key_version = fingerprint_secret_bytes(settings)
    verifier = fingerprint_key_material_verifier(secret.decode('utf-8'))
    for kind, knowledge_type, knowledge_id, declared_fingerprint in expected:
        link_statement = select(TrustedKnowledgeApprovalLink).where(
            TrustedKnowledgeApprovalLink.knowledge_type == knowledge_type,
            TrustedKnowledgeApprovalLink.knowledge_id == knowledge_id,
            TrustedKnowledgeApprovalLink.promotion_effect_kind == kind,
            TrustedKnowledgeApprovalLink.active.is_(True),
            TrustedKnowledgeApprovalLink.security_scope_id == scope,
            TrustedKnowledgeApprovalLink.permission_level == item.permission_level,
        )
        if db.get_bind().dialect.name == 'postgresql':
            link_statement = link_statement.with_for_update()
        links = tuple(db.scalars(link_statement).all())
        if not links:
            raise TrustedProvenanceMismatch('canonical approval provenance is missing')
        for link in links:
            current_link = (
                link.fingerprint_key_version == key_version
                and link.fingerprint_key_material_verifier == verifier
            )
            if current_link and link.claim_fingerprint != declared_fingerprint:
                raise TrustedProvenanceMismatch(
                    'canonical approval claim fingerprint is stale'
                )
            _verify_link_children(db, link=link, settings=settings)
        link_review_sets.append({link.review_item_id for link in links})
        target_effect = _effect_from_target(
            db,
            kind=kind,
            knowledge_type=knowledge_type,
            knowledge_id=knowledge_id,
            security_scope_id=scope,
            settings=settings,
            lock=True,
            expected_permission_level=item.permission_level,
        )
        if target_effect.claim_fingerprint != declared_fingerprint:
            raise TrustedProvenanceMismatch('canonical claim fingerprint is stale')
        effects.append(target_effect)

    if len(link_review_sets) == 2 and link_review_sets[0] != link_review_sets[1]:
        raise TrustedProvenanceMismatch(
            'canonical primary and companion origins are not exact'
        )

    candidate_fingerprints = [
        normalized_claim_fingerprint(
            item=item,
            security_scope_id=scope,
            settings=settings,
        )
    ]
    if directive.expected_type == 'history_event':
        preview = build_promotion_preview(item)
        normalized = preview['normalized_payload']
        candidate_fingerprints.append(
            promoted_effect_fingerprint(
                knowledge_type='timeline_event',
                normalized_persisted_fields={
                    'title': normalized['title'],
                    'result_summary': normalized['reason'],
                },
                project_key=(item.payload or {}).get('project_key'),
                security_scope_id=scope,
                settings=settings,
            )
        )
    if tuple(candidate_fingerprints) != tuple(
        effect.claim_fingerprint for effect in effects
    ):
        raise TrustedProvenanceMismatch('candidate does not exactly reaffirm the canonical bundle')
    return TrustedPromotionBundle(
        target_type=directive.expected_type,
        record_ids=() if directive.expected_type == 'timeline_event' else (directive.expected_id,),
        timeline_ids=(
            (directive.expected_id,)
            if directive.expected_type == 'timeline_event'
            else (directive.expected_companion_id,)  # type: ignore[arg-type]
        ),
        effects=tuple(effects),
    )


def has_legacy_human_base(
    db: Session, *, knowledge_type: str, knowledge_id: int
) -> bool:
    model = _model_for_type(knowledge_type)
    target = db.get(model, knowledge_id)
    if target is None or target.source_review_item_id is None:
        return False
    item = db.get(ReviewItem, target.source_review_item_id)
    return bool(
        item is not None
        and item.status == 'approved'
        and item.resolution_source in {None, 'human'}
        and item.candidate_contract_version != 'c5-v1'
    )


def replay_effect_for_bundle(
    db: Session,
    *,
    item: ReviewItem,
    bundle: TrustedPromotionBundle,
) -> Literal['created', 'reaffirmed']:
    primary = next(
        (effect for effect in bundle.effects if effect.kind == 'primary'),
        None,
    )
    if primary is None:
        raise TrustedProvenanceMismatch('explicit promotion primary is missing')
    target = db.get(_model_for_type(primary.knowledge_type), primary.knowledge_id)
    if target is None:
        raise TrustedProvenanceMismatch('explicit promotion target is missing')
    return 'created' if target.source_review_item_id == item.id else 'reaffirmed'


def _effect_from_target(
    db: Session,
    *,
    kind: str,
    knowledge_type: str,
    knowledge_id: int,
    security_scope_id: str,
    settings: Settings,
    lock: bool = False,
    expected_permission_level: str | None = None,
) -> TrustedPromotionEffect:
    model = _model_for_type(knowledge_type)
    statement = select(model).where(model.id == knowledge_id)
    if lock and db.get_bind().dialect.name == 'postgresql':
        statement = statement.with_for_update()
    target = db.scalar(statement)
    if target is None or target.review_status != 'approved':
        raise TrustedProvenanceMismatch('canonical trusted target is unavailable')
    if (
        expected_permission_level is not None
        and target.permission_level != expected_permission_level
    ):
        raise TrustedProvenanceMismatch('canonical target permission has changed')
    fields = _persisted_fields(knowledge_type, target)
    fingerprint = promoted_effect_fingerprint(
        knowledge_type=knowledge_type,  # type: ignore[arg-type]
        normalized_persisted_fields=fields,
        project_key=target.project_key,
        security_scope_id=security_scope_id,
        settings=settings,
    )
    return TrustedPromotionEffect(kind, knowledge_type, knowledge_id, fingerprint)


def _model_for_type(knowledge_type: str) -> type:
    try:
        return {
            'decision_record': DecisionRecord,
            'history_event': HistoryEvent,
            'timeline_event': TimelineEvent,
            'todo': Todo,
        }[knowledge_type]
    except KeyError:
        raise TrustedProvenanceMismatch('trusted target type is unsupported') from None


def _persisted_fields(knowledge_type: str, target: object) -> dict[str, str | None]:
    if knowledge_type == 'decision_record':
        return {'title': target.title, 'decision_summary': target.decision_summary}
    if knowledge_type == 'history_event':
        return {'title': target.title, 'reason': target.reason}
    if knowledge_type == 'timeline_event':
        return {'title': target.title, 'result_summary': target.result_summary}
    if knowledge_type == 'todo':
        return {
            'title': target.title,
            'assignee': target.assignee,
            'due_date': target.due_date,
            'priority': target.priority,
            'priority_reason': target.priority_reason,
        }
    raise TrustedProvenanceMismatch('trusted target type is unsupported')


def _security_scope(db: Session, item: ReviewItem) -> str:
    if item.workflow_thread_id is None:
        raise TrustedProvenanceMismatch('trusted provenance requires a security scope')
    workflow = db.get(AgentWorkflowThread, item.workflow_thread_id)
    if workflow is None or not workflow.security_scope_id:
        raise TrustedProvenanceMismatch('trusted provenance security scope is missing')
    return workflow.security_scope_id


def _parent_identity(link: TrustedKnowledgeApprovalLink) -> tuple[object, ...]:
    return (
        link.knowledge_type,
        link.knowledge_id,
        link.security_scope_id,
        link.resolution_source,
        link.claim_fingerprint,
        link.permission_level,
        link.fingerprint_key_version,
        link.fingerprint_key_material_verifier,
    )


def _evidence_identity(
    *,
    ref: AgentWorkflowEvidenceRef,
    binding: ReviewItemEvidenceRef,
    secret: bytes,
    key_version: str,
    verifier: str,
) -> tuple[str, str, str, str, str, str]:
    version = ref.external_revision or ref.content_signature
    evidence_hash = keyed_fingerprint(
        {
            'canonical_source_kind': ref.canonical_source_type,
            'canonical_table': ref.canonical_table,
            'canonical_source_id': str(ref.canonical_row_id),
            'canonical_version_or_signature': version,
            'document_version_id': ref.document_version_id,
            'content_signature': ref.content_signature,
            'content_fingerprint': ref.content_fingerprint,
            'permission_level_snapshot': ref.permission_level_snapshot,
            'workflow_evidence_ordinal': ref.ordinal,
            'workflow_evidence_ref_id': ref.id,
            'candidate_slot_ordinal': binding.candidate_slot_ordinal,
            'review_item_id': binding.review_item_id,
            'message_content_fingerprint': binding.message_content_fingerprint,
        },
        secret=secret,
        schema_version='trusted-evidence-provenance:v1',
        policy_version='trusted-evidence-provenance:v1',
    )
    return (
        ref.canonical_source_type,
        str(ref.canonical_row_id),
        version,
        evidence_hash,
        key_version,
        verifier,
    )


def _child_identity(child: TrustedKnowledgeEvidenceLink) -> tuple[str, str, str, str, str, str]:
    return (
        child.canonical_source_kind,
        child.canonical_source_id,
        child.canonical_version_or_signature,
        child.evidence_hash,
        child.fingerprint_key_version,
        child.fingerprint_key_material_verifier,
    )


def _verify_link_children(
    db: Session,
    *,
    link: TrustedKnowledgeApprovalLink,
    settings: Settings,
) -> None:
    item = db.get(ReviewItem, link.review_item_id)
    if (
        item is None
        or item.workflow_thread_id is None
        or item.status != 'approved'
        or item.resolution_source != link.resolution_source
    ):
        raise TrustedProvenanceMismatch('approval provenance owner is missing')
    bindings = tuple(
        db.scalars(
            select(ReviewItemEvidenceRef)
            .where(ReviewItemEvidenceRef.review_item_id == item.id)
            .order_by(ReviewItemEvidenceRef.candidate_slot_ordinal)
        ).all()
    )
    refs = {
        ref.id: ref
        for ref in db.scalars(
            select(AgentWorkflowEvidenceRef).where(
                AgentWorkflowEvidenceRef.id.in_(
                    [binding.workflow_evidence_ref_id for binding in bindings]
                )
            )
        ).all()
    }
    children = tuple(
        db.scalars(
            select(TrustedKnowledgeEvidenceLink).where(
                TrustedKnowledgeEvidenceLink.approval_link_id == link.id
            )
        ).all()
    )
    if not bindings or len(refs) != len(bindings) or len(children) != len(bindings):
        raise TrustedProvenanceMismatch('approval evidence provenance is incomplete')
    if any(
        child.fingerprint_key_version != link.fingerprint_key_version
        or child.fingerprint_key_material_verifier
        != link.fingerprint_key_material_verifier
        for child in children
    ):
        raise TrustedProvenanceMismatch('approval evidence key identity is inconsistent')
    expected_base = sorted(
        (
            refs[binding.workflow_evidence_ref_id].canonical_source_type,
            str(refs[binding.workflow_evidence_ref_id].canonical_row_id),
            refs[binding.workflow_evidence_ref_id].external_revision
            or refs[binding.workflow_evidence_ref_id].content_signature,
        )
        for binding in bindings
    )
    actual_base = sorted(
        (
            child.canonical_source_kind,
            child.canonical_source_id,
            child.canonical_version_or_signature,
        )
        for child in children
    )
    if (
        actual_base != expected_base
        or len({child.evidence_hash for child in children}) != len(children)
    ):
        raise TrustedProvenanceMismatch('approval evidence identities are inconsistent')
    secret, key_version = fingerprint_secret_bytes(settings)
    verifier = fingerprint_key_material_verifier(secret.decode('utf-8'))
    if (
        link.fingerprint_key_version == key_version
        and link.fingerprint_key_material_verifier == verifier
    ):
        expected = {
            _evidence_identity(
                ref=refs[binding.workflow_evidence_ref_id],
                binding=binding,
                secret=secret,
                key_version=key_version,
                verifier=verifier,
            )
            for binding in bindings
        }
        if (
            len(expected) != len(bindings)
            or {_child_identity(child) for child in children} != expected
        ):
            raise TrustedProvenanceMismatch(
                'approval evidence HMAC set is inconsistent'
            )
