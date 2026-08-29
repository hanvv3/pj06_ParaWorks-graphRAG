from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Integer,
    Select,
    String,
    and_,
    bindparam,
    column,
    delete,
    exists,
    func,
    or_,
    select,
    tuple_,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.agent_runtime.keyed_mutation_guard import (
    KeyedMutationGuard,
    acquire_projection,
    lock_runtime_state,
)
from backend.app.core.config import Settings
from backend.app.knowledge.claim_fingerprints import (
    promoted_effect_fingerprint,
    trusted_project_scope_fingerprint,
    trusted_title_collision_bucket,
)
from backend.app.models import (
    AgentWorkflowThread,
    AutoReviewExtractionCall,
    AutoReviewRuntimeKeyState,
    AutoReviewValidationCall,
    HistoryEvent,
    ReviewItem,
    TimelineEvent,
    TrustedKnowledgeFingerprint,
    TrustedKnowledgeFingerprintProjectionState,
)

if TYPE_CHECKING:
    from backend.app.admin.auto_review_keys import (
        FingerprintKeyRing,
        ProjectionRebuildResult,
    )

TRUSTED_FINGERPRINT_PROJECTION_COMPONENT = 'trusted_knowledge_fingerprints'
TRUSTED_FINGERPRINT_PROJECTION_SCHEMA = 'trusted-fingerprint-projection:v1'
_MODULUS = 1 << 256
_ZERO_CHECKSUM = '0' * 64


@dataclass(frozen=True, slots=True)
class ProjectionSummary:
    active_count: int
    checksum_hex: str

    def __post_init__(self) -> None:
        if self.active_count < 0:
            raise ValueError('projection summary count cannot be negative')
        _checksum_int(self.checksum_hex)


@dataclass(frozen=True, slots=True)
class ProjectionSummaryDelta:
    old_digest: str | None
    new_digest: str | None

    def apply(self, summary: ProjectionSummary) -> ProjectionSummary:
        count = summary.active_count
        checksum = _checksum_int(summary.checksum_hex)
        if self.old_digest is not None:
            checksum = (checksum - _checksum_int(self.old_digest)) % _MODULUS
            count -= 1
        if self.new_digest is not None:
            checksum = (checksum + _checksum_int(self.new_digest)) % _MODULUS
            count += 1
        if count < 0:
            raise ValueError('projection summary delta removes a missing row')
        return ProjectionSummary(count, f'{checksum:064x}')


@dataclass(frozen=True, slots=True)
class ProjectionRowSnapshot:
    knowledge_type: str
    knowledge_id: int
    scope_resolution: str
    security_scope_id: str | None
    project_scope_hmac: str
    normalized_title_bucket_hmac: str
    normalized_claim_fingerprint: str | None
    permission_level: str
    review_status: str
    fingerprint_key_version: str
    fingerprint_key_material_verifier: str

    def __post_init__(self) -> None:
        if self.knowledge_type not in {'timeline_event', 'history_event'}:
            raise ValueError('projection supports Timeline and History only')
        if self.knowledge_id <= 0:
            raise ValueError('projection knowledge id must be positive')
        if self.scope_resolution not in {'exact', 'legacy_unknown'}:
            raise ValueError('projection scope resolution is invalid')
        if self.scope_resolution == 'exact' and not self.security_scope_id:
            raise ValueError('exact projection scope requires an id')
        if self.scope_resolution == 'legacy_unknown' and (
            self.security_scope_id is not None
            or self.normalized_claim_fingerprint is not None
        ):
            raise ValueError('legacy-unknown projection is collision-only')


def projection_row_digest(
    snapshot: ProjectionRowSnapshot,
    *,
    settings: Settings,
) -> str:
    secret, configured_version = fingerprint_secret_bytes(settings)
    if snapshot.fingerprint_key_version != configured_version:
        raise ValueError('projection row key version does not match settings')
    return keyed_fingerprint(
        {
            'projection_schema_version': TRUSTED_FINGERPRINT_PROJECTION_SCHEMA,
            'knowledge_type': snapshot.knowledge_type,
            'knowledge_id': snapshot.knowledge_id,
            'scope_resolution': snapshot.scope_resolution,
            'security_scope_id': snapshot.security_scope_id,
            'project_scope_hmac': snapshot.project_scope_hmac,
            'normalized_title_bucket_hmac': snapshot.normalized_title_bucket_hmac,
            'normalized_claim_fingerprint': snapshot.normalized_claim_fingerprint,
            'permission_level': snapshot.permission_level,
            'review_status': snapshot.review_status,
            'fingerprint_key_version': snapshot.fingerprint_key_version,
            'fingerprint_key_material_verifier': (
                snapshot.fingerprint_key_material_verifier
            ),
        },
        secret=secret,
        schema_version=TRUSTED_FINGERPRINT_PROJECTION_SCHEMA,
        policy_version='trusted-fingerprint-summary-row:v1',
    )


def build_hidden_collision_exists_statement(
    *,
    knowledge_type: str,
    security_scope_id: str,
    project_scope_hmac: str,
    normalized_title_bucket_hmac: str,
    visible_permission_levels: tuple[str, ...],
    candidate_permission_level: str | None = None,
) -> Select:
    if knowledge_type not in {'timeline_event', 'history_event'}:
        raise ValueError('hidden collision lookup type is unsupported')
    bucket = and_(
        TrustedKnowledgeFingerprint.knowledge_type == knowledge_type,
        TrustedKnowledgeFingerprint.project_scope_hmac == project_scope_hmac,
        TrustedKnowledgeFingerprint.normalized_title_bucket_hmac
        == normalized_title_bucket_hmac,
        TrustedKnowledgeFingerprint.review_status == 'approved',
    )
    if visible_permission_levels:
        collision = and_(
            bucket,
            or_(
                TrustedKnowledgeFingerprint.scope_resolution == 'legacy_unknown',
                and_(
                    TrustedKnowledgeFingerprint.scope_resolution == 'exact',
                    TrustedKnowledgeFingerprint.security_scope_id
                    == security_scope_id,
                    or_(
                        TrustedKnowledgeFingerprint.permission_level.not_in(
                            visible_permission_levels
                        ),
                        *(
                            (
                                TrustedKnowledgeFingerprint.permission_level
                                != candidate_permission_level,
                            )
                            if candidate_permission_level is not None
                            else ()
                        ),
                    ),
                ),
            ),
        )
    else:
        collision = and_(
            bucket,
            or_(
                TrustedKnowledgeFingerprint.scope_resolution == 'legacy_unknown',
                and_(
                    TrustedKnowledgeFingerprint.scope_resolution == 'exact',
                    TrustedKnowledgeFingerprint.security_scope_id
                    == security_scope_id,
                ),
            ),
        )
    return select(exists().where(collision))


@dataclass(frozen=True, slots=True)
class VisibleTrustedCollision:
    knowledge_type: str
    knowledge_id: int
    normalized_claim_fingerprint: str


def find_visible_trusted_collisions(
    db: Session,
    *,
    knowledge_type: str,
    security_scope_id: str,
    project_scope_hmac: str,
    normalized_title_bucket_hmac: str,
    visible_permission_levels: tuple[str, ...],
    candidate_permission_level: str,
) -> tuple[VisibleTrustedCollision, ...]:
    if knowledge_type not in {'timeline_event', 'history_event'}:
        raise ValueError('visible collision lookup type is unsupported')
    if not visible_permission_levels:
        return ()
    rows = tuple(
        db.execute(
            select(
                TrustedKnowledgeFingerprint.knowledge_type,
                TrustedKnowledgeFingerprint.knowledge_id,
                TrustedKnowledgeFingerprint.normalized_claim_fingerprint,
            )
            .where(
                TrustedKnowledgeFingerprint.knowledge_type == knowledge_type,
                TrustedKnowledgeFingerprint.scope_resolution == 'exact',
                TrustedKnowledgeFingerprint.security_scope_id == security_scope_id,
                TrustedKnowledgeFingerprint.project_scope_hmac == project_scope_hmac,
                TrustedKnowledgeFingerprint.normalized_title_bucket_hmac
                == normalized_title_bucket_hmac,
                TrustedKnowledgeFingerprint.review_status == 'approved',
                TrustedKnowledgeFingerprint.permission_level.in_(
                    visible_permission_levels
                ),
                TrustedKnowledgeFingerprint.permission_level
                == candidate_permission_level,
                TrustedKnowledgeFingerprint.normalized_claim_fingerprint.is_not(None),
            )
            .order_by(TrustedKnowledgeFingerprint.knowledge_id)
            .limit(3)
        ).all()
    )
    return tuple(
        VisibleTrustedCollision(
            knowledge_type=row.knowledge_type,
            knowledge_id=row.knowledge_id,
            normalized_claim_fingerprint=row.normalized_claim_fingerprint,
        )
        for row in rows
    )


def hidden_or_legacy_collision_exists(
    db: Session,
    *,
    knowledge_type: str,
    security_scope_id: str,
    project_scope_hmac: str,
    normalized_title_bucket_hmac: str,
    visible_permission_levels: tuple[str, ...],
    candidate_permission_level: str,
) -> bool:
    return bool(
        db.scalar(
            build_hidden_collision_exists_statement(
                knowledge_type=knowledge_type,
                security_scope_id=security_scope_id,
                project_scope_hmac=project_scope_hmac,
                normalized_title_bucket_hmac=normalized_title_bucket_hmac,
                visible_permission_levels=visible_permission_levels,
                candidate_permission_level=candidate_permission_level,
            )
        )
    )


def build_missing_active_projection_exists_statement(
    *,
    expected_rows: tuple[ProjectionRowSnapshot, ...],
    excluded_targets: tuple[tuple[str, int], ...] = (),
) -> Select:
    excluded = frozenset(excluded_targets)
    expected = tuple(
        row
        for row in expected_rows
        if (row.knowledge_type, row.knowledge_id) not in excluded
    )

    projection_in_scope: object = True
    if excluded:
        projection_in_scope = tuple_(
            TrustedKnowledgeFingerprint.knowledge_type,
            TrustedKnowledgeFingerprint.knowledge_id,
        ).not_in(tuple(sorted(excluded)))
    if not expected:
        return select(
            select(TrustedKnowledgeFingerprint.id)
            .where(projection_in_scope)
            .exists()
        )
    expected_payload = [
        {
            field: getattr(row, field)
            for field in ProjectionRowSnapshot.__dataclass_fields__
        }
        for row in expected
    ]
    expected_values = (
        func.jsonb_to_recordset(
            bindparam(
                'expected_projection_rows',
                value=expected_payload,
                type_=JSONB,
            )
        )
        .table_valued(
            column('knowledge_type', String()),
            column('knowledge_id', Integer()),
            column('scope_resolution', String()),
            column('security_scope_id', String()),
            column('project_scope_hmac', String()),
            column('normalized_title_bucket_hmac', String()),
            column('normalized_claim_fingerprint', String()),
            column('permission_level', String()),
            column('review_status', String()),
            column('fingerprint_key_version', String()),
            column('fingerprint_key_material_verifier', String()),
        )
        .render_derived(
            name='expected_trusted_fingerprints',
            with_types=True,
        )
    )
    exact_match = and_(
        TrustedKnowledgeFingerprint.knowledge_type
        == expected_values.c.knowledge_type,
        TrustedKnowledgeFingerprint.knowledge_id == expected_values.c.knowledge_id,
        TrustedKnowledgeFingerprint.scope_resolution
        == expected_values.c.scope_resolution,
        TrustedKnowledgeFingerprint.security_scope_id.is_not_distinct_from(
            expected_values.c.security_scope_id
        ),
        TrustedKnowledgeFingerprint.project_scope_hmac
        == expected_values.c.project_scope_hmac,
        TrustedKnowledgeFingerprint.normalized_title_bucket_hmac
        == expected_values.c.normalized_title_bucket_hmac,
        TrustedKnowledgeFingerprint.normalized_claim_fingerprint.is_not_distinct_from(
            expected_values.c.normalized_claim_fingerprint
        ),
        TrustedKnowledgeFingerprint.permission_level
        == expected_values.c.permission_level,
        TrustedKnowledgeFingerprint.review_status == expected_values.c.review_status,
        TrustedKnowledgeFingerprint.fingerprint_key_version
        == expected_values.c.fingerprint_key_version,
        TrustedKnowledgeFingerprint.fingerprint_key_material_verifier
        == expected_values.c.fingerprint_key_material_verifier,
    )
    missing = (
        select(expected_values.c.knowledge_id)
        .where(~select(TrustedKnowledgeFingerprint.id).where(exact_match).exists())
        .exists()
    )
    extra = (
        select(TrustedKnowledgeFingerprint.id)
        .where(
            projection_in_scope,
            ~select(expected_values.c.knowledge_id).where(exact_match).exists(),
        )
        .exists()
    )
    return select(or_(missing, extra))


def source_projection_snapshots(
    db: Session,
    *,
    settings: Settings,
    fingerprint_key_material_verifier: str,
) -> tuple[ProjectionRowSnapshot, ...]:
    return _source_snapshots(
        db,
        settings=settings,
        verifier=fingerprint_key_material_verifier,
    )


def projection_identity_ready(
    runtime: AutoReviewRuntimeKeyState | None,
    projection: TrustedKnowledgeFingerprintProjectionState | None,
    *,
    missing_active_row: bool,
) -> bool:
    if runtime is None or projection is None or missing_active_row:
        return False
    return bool(
        runtime.ready
        and projection.ready
        and not projection.rebuild_required
        and projection.projection_schema_version
        == TRUSTED_FINGERPRINT_PROJECTION_SCHEMA
        and projection.fingerprint_key_version
        == runtime.fingerprint_key_version
        and projection.fingerprint_key_material_verifier
        == runtime.fingerprint_key_material_verifier
        and projection.generation == runtime.generation
        and projection.source_active_count == projection.projected_active_count
        and (projection.source_checksum or _ZERO_CHECKSUM)
        == (projection.projected_checksum or _ZERO_CHECKSUM)
    )


def trusted_fingerprint_projection_ready(
    db: Session,
    *,
    settings: Settings,
) -> bool:
    """Return server-owned readiness for exact trusted-collision lookups."""
    if db.get_bind().dialect.name != 'postgresql':
        return False
    try:
        runtime = _runtime_state(db)
        projection = _projection_state(db)
        verifier = fingerprint_key_material_verifier(
            settings.agent_runtime_fingerprint_secret
        )
        if (
            runtime is None
            or runtime.fingerprint_key_version
            != settings.agent_runtime_fingerprint_key_version
            or runtime.fingerprint_key_material_verifier != verifier
        ):
            return False
        expected_rows = source_projection_snapshots(
            db,
            settings=settings,
            fingerprint_key_material_verifier=verifier,
        )
        missing_active_row = bool(
            db.scalar(
                build_missing_active_projection_exists_statement(
                    expected_rows=expected_rows,
                )
            )
        )
        return projection_identity_ready(
            runtime,
            projection,
            missing_active_row=missing_active_row,
        )
    except (SQLAlchemyError, TypeError, ValueError):
        return False


def snapshot_from_projection(
    row: TrustedKnowledgeFingerprint,
) -> ProjectionRowSnapshot:
    return ProjectionRowSnapshot(
        knowledge_type=row.knowledge_type,
        knowledge_id=row.knowledge_id,
        scope_resolution=row.scope_resolution,
        security_scope_id=row.security_scope_id,
        project_scope_hmac=row.project_scope_hmac,
        normalized_title_bucket_hmac=row.normalized_title_bucket_hmac,
        normalized_claim_fingerprint=row.normalized_claim_fingerprint,
        permission_level=row.permission_level,
        review_status=row.review_status,
        fingerprint_key_version=row.fingerprint_key_version,
        fingerprint_key_material_verifier=(
            row.fingerprint_key_material_verifier
        ),
    )


def summary_for_rows(
    rows: tuple[ProjectionRowSnapshot, ...],
    *,
    settings: Settings,
) -> ProjectionSummary:
    summary = ProjectionSummary(0, _ZERO_CHECKSUM)
    seen: set[tuple[str, int]] = set()
    for row in rows:
        identity = (row.knowledge_type, row.knowledge_id)
        if identity in seen:
            raise ValueError('projection summary contains duplicate target rows')
        seen.add(identity)
        summary = ProjectionSummaryDelta(
            old_digest=None,
            new_digest=projection_row_digest(row, settings=settings),
        ).apply(summary)
    return summary


def project_approved_effects(
    db: Session,
    *,
    effects: tuple[object, ...],
    settings: Settings,
    new_target_effects: bool = False,
) -> None:
    """Apply exact Timeline/History rows without ever blocking human approval."""
    runtime = _runtime_state(db)
    state = _projection_state(db)
    if state is None:
        return
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    excluded_targets = (
        tuple((effect.knowledge_type, effect.knowledge_id) for effect in effects)
        if new_target_effects
        else ()
    )
    missing_active_row = True
    if runtime is not None and db.get_bind().dialect.name == 'postgresql':
        try:
            expected_rows = source_projection_snapshots(
                db,
                settings=settings,
                fingerprint_key_material_verifier=(
                    runtime.fingerprint_key_material_verifier
                ),
            )
            missing_active_row = bool(
                db.scalar(
                    build_missing_active_projection_exists_statement(
                        expected_rows=expected_rows,
                        excluded_targets=excluded_targets,
                    )
                )
            )
        except (TypeError, ValueError):
            missing_active_row = True
    started_healthy = bool(
        db.get_bind().dialect.name == 'postgresql'
        and projection_identity_ready(
            runtime,
            state,
            missing_active_row=missing_active_row,
        )
        and runtime is not None
        and runtime.fingerprint_key_version
        == settings.agent_runtime_fingerprint_key_version
        and runtime.fingerprint_key_material_verifier == verifier
    )
    try:
        for effect in effects:
            knowledge_type = effect.knowledge_type
            if knowledge_type not in {'timeline_event', 'history_event'}:
                continue
            model = HistoryEvent if knowledge_type == 'history_event' else TimelineEvent
            target = db.get(model, effect.knowledge_id)
            if target is None:
                raise ValueError('projectable trusted target is missing')
            scope = _target_scope(db, target.source_review_item_id)
            if scope is None:
                raise ValueError('new trusted target scope is missing')
            snapshot = ProjectionRowSnapshot(
                knowledge_type=knowledge_type,
                knowledge_id=target.id,
                scope_resolution='exact',
                security_scope_id=scope,
                project_scope_hmac=trusted_project_scope_fingerprint(
                    project_key=target.project_key, settings=settings
                ),
                normalized_title_bucket_hmac=trusted_title_collision_bucket(
                    item_type=knowledge_type,
                    normalized_title=target.title,
                    settings=settings,
                ),
                normalized_claim_fingerprint=effect.claim_fingerprint,
                permission_level=target.permission_level,
                review_status=target.review_status,
                fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
                fingerprint_key_material_verifier=verifier,
            )
            row = db.scalar(
                select(TrustedKnowledgeFingerprint).where(
                    TrustedKnowledgeFingerprint.knowledge_type == knowledge_type,
                    TrustedKnowledgeFingerprint.knowledge_id == target.id,
                )
            )
            old_digest = (
                projection_row_digest(snapshot_from_projection(row), settings=settings)
                if row is not None
                else None
            )
            if row is None:
                row = _row_from_snapshot(snapshot)
                db.add(row)
            else:
                for field in ProjectionRowSnapshot.__dataclass_fields__:
                    if field not in {'knowledge_type', 'knowledge_id'}:
                        setattr(row, field, getattr(snapshot, field))
                row.updated_at = datetime.now(UTC)
            new_digest = projection_row_digest(snapshot, settings=settings)
            if started_healthy:
                source = ProjectionSummary(
                    state.source_active_count,
                    state.source_checksum or _ZERO_CHECKSUM,
                )
                projected = ProjectionSummary(
                    state.projected_active_count,
                    state.projected_checksum or _ZERO_CHECKSUM,
                )
                delta = ProjectionSummaryDelta(old_digest, new_digest)
                source = delta.apply(source)
                projected = delta.apply(projected)
                state.source_active_count = source.active_count
                state.source_checksum = source.checksum_hex
                state.projected_active_count = projected.active_count
                state.projected_checksum = projected.checksum_hex
        if not started_healthy:
            state.ready = False
            state.rebuild_required = True
            state.completed_at = None
        state.updated_at = datetime.now(UTC)
        db.flush()
    except (TypeError, ValueError):
        state.ready = False
        state.rebuild_required = True
        state.completed_at = None
        state.updated_at = datetime.now(UTC)


def demote_trusted_fingerprint_projection(db: Session) -> None:
    """Mark the projection stale after an unprojectable legacy approval."""
    state = _projection_state(db)
    if state is None:
        return
    state.ready = False
    state.rebuild_required = True
    state.completed_at = None
    state.updated_at = datetime.now(UTC)
    db.flush([state])


def rebuild_trusted_fingerprint_projection(
    *,
    session_factory: sessionmaker,
    settings: Settings,
) -> ProjectionRebuildResult:
    from backend.app.admin.auto_review_keys import (
        AutoReviewKeyAdminError,
        ProjectionRebuildResult,
    )

    with session_factory() as db, KeyedMutationGuard.generation_barrier(db):
        context = lock_runtime_state(db, mode='share')
        if context is None:
            raise AutoReviewKeyAdminError(
                'runtime_key_state_missing', 'runtime key state is not initialized'
            )
        acquire_projection(db, context)
        state = _projection_state(db)
        if state is None:
            raise AutoReviewKeyAdminError(
                'projection_state_missing', 'projection state is not initialized'
            )
        state.ready = False
        state.rebuild_required = True
        state.updated_at = datetime.now(UTC)
        db.commit()

    with session_factory() as db, KeyedMutationGuard.generation_barrier(db):
        context = lock_runtime_state(db, mode='share')
        if context is None:
            raise AutoReviewKeyAdminError(
                'runtime_key_state_missing', 'runtime key state is not initialized'
            )
        acquire_projection(db, context)
        runtime = _runtime_state(db)
        state = _projection_state(db)
        if runtime is None or state is None:
            raise AutoReviewKeyAdminError(
                'key_state_missing', 'runtime or projection state disappeared'
            )
        configured_secret, configured_version = fingerprint_secret_bytes(settings)
        configured_verifier = fingerprint_key_material_verifier(
            configured_secret.decode('utf-8')
        )
        if (
            runtime.fingerprint_key_version != configured_version
            or runtime.fingerprint_key_material_verifier != configured_verifier
        ):
            raise AutoReviewKeyAdminError(
                'runtime_key_identity_mismatch',
                'configured key identity does not match runtime state',
            )
        old_rows = tuple(
            snapshot_from_projection(row)
            for row in db.scalars(
                select(TrustedKnowledgeFingerprint).order_by(
                    TrustedKnowledgeFingerprint.knowledge_type,
                    TrustedKnowledgeFingerprint.knowledge_id,
                )
            ).all()
        )
        expected = source_projection_snapshots(
            db,
            settings=settings,
            fingerprint_key_material_verifier=configured_verifier,
        )
        expected_summary = summary_for_rows(expected, settings=settings)
        replayed = old_rows == expected
        db.execute(delete(TrustedKnowledgeFingerprint))
        db.add_all([_row_from_snapshot(snapshot) for snapshot in expected])
        db.flush()
        projected = tuple(
            snapshot_from_projection(row)
            for row in db.scalars(
                select(TrustedKnowledgeFingerprint).order_by(
                    TrustedKnowledgeFingerprint.knowledge_type,
                    TrustedKnowledgeFingerprint.knowledge_id,
                )
            ).all()
        )
        projected_summary = summary_for_rows(projected, settings=settings)
        missing_active_row = True
        if db.get_bind().dialect.name == 'postgresql':
            missing_active_row = bool(
                db.scalar(
                    build_missing_active_projection_exists_statement(
                        expected_rows=expected,
                    )
                )
            )
        ready = bool(
            db.get_bind().dialect.name == 'postgresql'
            and runtime.ready
            and expected == projected
            and expected_summary == projected_summary
            and not missing_active_row
        )
        state.projection_schema_version = TRUSTED_FINGERPRINT_PROJECTION_SCHEMA
        state.fingerprint_key_version = runtime.fingerprint_key_version
        state.fingerprint_key_material_verifier = (
            runtime.fingerprint_key_material_verifier
        )
        state.generation = runtime.generation
        state.source_active_count = expected_summary.active_count
        state.projected_active_count = projected_summary.active_count
        state.source_checksum = expected_summary.checksum_hex
        state.projected_checksum = projected_summary.checksum_hex
        state.ready = ready
        state.rebuild_required = not ready
        state.completed_at = datetime.now(UTC) if ready else None
        state.updated_at = datetime.now(UTC)
        db.commit()
        return ProjectionRebuildResult(
            source_count=expected_summary.active_count,
            projected_count=projected_summary.active_count,
            source_checksum=expected_summary.checksum_hex,
            projected_checksum=projected_summary.checksum_hex,
            replayed=replayed,
            ready=ready,
        )


def rotate_fingerprint_key(
    *,
    session_factory: sessionmaker,
    settings: Settings,
    key_ring: FingerprintKeyRing,
    expected_version: str,
    next_version: str,
    reason: str,
    principal: str,
) -> ProjectionRebuildResult:
    from backend.app.admin.auto_review_keys import AutoReviewKeyAdminError

    if settings.auto_review_mode != 'disabled':
        raise AutoReviewKeyAdminError(
            'auto_review_not_disabled', 'key rotation requires disabled mode'
        )
    probe = session_factory()
    try:
        if probe.get_bind().dialect.name != 'postgresql':
            raise AutoReviewKeyAdminError(
                'postgresql_required', 'durable key rotation requires PostgreSQL'
            )
    finally:
        probe.close()
    if principal != 'system:local-auto-review-key-admin':
        raise AutoReviewKeyAdminError(
            'operator_identity_invalid', 'key rotation requires the local operator'
        )
    if not 1 <= len(reason.strip()) <= 500:
        raise AutoReviewKeyAdminError('reason_invalid', 'rotation reason is invalid')
    if (
        key_ring.current_version != expected_version
        or key_ring.next_version != next_version
        or expected_version == next_version
    ):
        raise AutoReviewKeyAdminError(
            'key_version_mismatch', 'key-ring versions do not match the request'
        )
    current_secret = key_ring.current_secret.get_secret_value()
    next_secret = key_ring.next_secret.get_secret_value()
    current_verifier = fingerprint_key_material_verifier(current_secret)
    next_verifier = fingerprint_key_material_verifier(next_secret)
    if current_verifier == next_verifier:
        raise AutoReviewKeyAdminError(
            'key_material_unchanged', 'next key material must differ'
        )

    with session_factory() as db, KeyedMutationGuard.generation_barrier(
        db, exclusive=True
    ):
        context = lock_runtime_state(db, mode='update')
        if context is None:
            raise AutoReviewKeyAdminError(
                'runtime_key_state_missing', 'runtime key state is not initialized'
            )
        runtime = _runtime_state(db)
        state = _projection_state(db)
        if runtime is None or state is None:
            raise AutoReviewKeyAdminError(
                'key_state_missing', 'runtime or projection state is missing'
            )
        if (
            runtime.fingerprint_key_version != expected_version
            or runtime.fingerprint_key_material_verifier != current_verifier
        ):
            raise AutoReviewKeyAdminError(
                'old_key_identity_mismatch', 'old key identity does not match runtime state'
            )
        extraction = tuple(
            db.scalars(
                build_nonterminal_provider_call_lock_statement(
                    AutoReviewExtractionCall
                )
            ).all()
        )
        validation = tuple(
            db.scalars(
                build_nonterminal_provider_call_lock_statement(
                    AutoReviewValidationCall
                )
            ).all()
        )
        if extraction or validation:
            raise AutoReviewKeyAdminError(
                'nonterminal_provider_calls',
                'provider call ledger recovery is required before key rotation',
            )
        acquire_projection(db, context)
        runtime.fingerprint_key_version = next_version
        runtime.fingerprint_key_material_verifier = next_verifier
        runtime.generation += 1
        runtime.ready = True
        runtime.updated_at = datetime.now(UTC)
        db.execute(delete(TrustedKnowledgeFingerprint))
        state.projection_schema_version = TRUSTED_FINGERPRINT_PROJECTION_SCHEMA
        state.fingerprint_key_version = next_version
        state.fingerprint_key_material_verifier = next_verifier
        state.generation = runtime.generation
        state.ready = False
        state.rebuild_required = True
        state.source_active_count = 0
        state.projected_active_count = 0
        state.source_checksum = _ZERO_CHECKSUM
        state.projected_checksum = _ZERO_CHECKSUM
        state.completed_at = None
        state.updated_at = datetime.now(UTC)
        db.commit()

    next_settings = settings.model_copy(
        update={
            'agent_runtime_fingerprint_key_version': next_version,
            'agent_runtime_fingerprint_secret': next_secret,
        }
    )
    return rebuild_trusted_fingerprint_projection(
        session_factory=session_factory,
        settings=next_settings,
    )


def build_nonterminal_provider_call_lock_statement(
    model: type[AutoReviewExtractionCall] | type[AutoReviewValidationCall],
) -> Select:
    if model not in {AutoReviewExtractionCall, AutoReviewValidationCall}:
        raise TypeError('provider call lock model is unsupported')
    return (
        select(model)
        .where(model.status == 'claimed')
        .order_by(model.id)
        .with_for_update()
    )


def _source_snapshots(
    db: Session,
    *,
    settings: Settings,
    verifier: str,
) -> tuple[ProjectionRowSnapshot, ...]:
    snapshots: list[ProjectionRowSnapshot] = []
    key_version = settings.agent_runtime_fingerprint_key_version.strip()
    for knowledge_type, model in (
        ('history_event', HistoryEvent),
        ('timeline_event', TimelineEvent),
    ):
        targets = tuple(
            db.execute(
                select(model, AgentWorkflowThread.security_scope_id)
                .outerjoin(ReviewItem, ReviewItem.id == model.source_review_item_id)
                .outerjoin(
                    AgentWorkflowThread,
                    AgentWorkflowThread.thread_id == ReviewItem.workflow_thread_id,
                )
                .where(model.review_status == 'approved')
                .order_by(model.id)
            ).all()
        )
        for target, scope in targets:
            exact = scope is not None
            fields = (
                {'title': target.title, 'reason': target.reason}
                if knowledge_type == 'history_event'
                else {'title': target.title, 'result_summary': target.result_summary}
            )
            snapshots.append(
                ProjectionRowSnapshot(
                    knowledge_type=knowledge_type,
                    knowledge_id=target.id,
                    scope_resolution='exact' if exact else 'legacy_unknown',
                    security_scope_id=scope,
                    project_scope_hmac=trusted_project_scope_fingerprint(
                        project_key=target.project_key, settings=settings
                    ),
                    normalized_title_bucket_hmac=trusted_title_collision_bucket(
                        item_type=knowledge_type,
                        normalized_title=target.title,
                        settings=settings,
                    ),
                    normalized_claim_fingerprint=(
                        promoted_effect_fingerprint(
                            knowledge_type=knowledge_type,
                            normalized_persisted_fields=fields,
                            project_key=target.project_key,
                            security_scope_id=scope,
                            settings=settings,
                        )
                        if exact
                        else None
                    ),
                    permission_level=target.permission_level,
                    review_status=target.review_status,
                    fingerprint_key_version=key_version,
                    fingerprint_key_material_verifier=verifier,
                )
            )
    return tuple(sorted(snapshots, key=lambda row: (row.knowledge_type, row.knowledge_id)))


def _target_scope(db: Session, review_item_id: int | None) -> str | None:
    if review_item_id is None:
        return None
    item = db.get(ReviewItem, review_item_id)
    if item is None or item.workflow_thread_id is None:
        return None
    workflow = db.get(AgentWorkflowThread, item.workflow_thread_id)
    return workflow.security_scope_id if workflow is not None else None


def _row_from_snapshot(snapshot: ProjectionRowSnapshot) -> TrustedKnowledgeFingerprint:
    return TrustedKnowledgeFingerprint(
        knowledge_type=snapshot.knowledge_type,
        knowledge_id=snapshot.knowledge_id,
        scope_resolution=snapshot.scope_resolution,
        security_scope_id=snapshot.security_scope_id,
        project_scope_hmac=snapshot.project_scope_hmac,
        normalized_title_bucket_hmac=snapshot.normalized_title_bucket_hmac,
        normalized_claim_fingerprint=snapshot.normalized_claim_fingerprint,
        permission_level=snapshot.permission_level,
        review_status=snapshot.review_status,
        fingerprint_key_version=snapshot.fingerprint_key_version,
        fingerprint_key_material_verifier=snapshot.fingerprint_key_material_verifier,
    )


def _runtime_state(db: Session) -> AutoReviewRuntimeKeyState | None:
    return db.scalar(
        select(AutoReviewRuntimeKeyState).where(
            AutoReviewRuntimeKeyState.component == 'auto_review_trust_promotion'
        )
    )


def _projection_state(
    db: Session,
) -> TrustedKnowledgeFingerprintProjectionState | None:
    return db.scalar(
        select(TrustedKnowledgeFingerprintProjectionState).where(
            TrustedKnowledgeFingerprintProjectionState.component
            == TRUSTED_FINGERPRINT_PROJECTION_COMPONENT
        )
    )


def _checksum_int(value: str) -> int:
    if len(value) != 64 or value.lower() != value:
        raise ValueError('projection checksum must be 64 lowercase hex characters')
    try:
        return int(value, 16)
    except ValueError:
        raise ValueError(
            'projection checksum must be 64 lowercase hex characters'
        ) from None
