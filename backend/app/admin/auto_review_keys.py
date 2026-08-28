import hashlib
import hmac
from dataclasses import dataclass

from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session, sessionmaker

from backend.app.agent_runtime.keyed_mutation_guard import KeyedMutationGuard
from backend.app.core.config import Settings
from backend.app.models.auto_review import (
    AutoReviewRuntimeKeyState,
    TrustedKnowledgeFingerprintProjectionState,
)

AUTO_REVIEW_KEY_COMPONENT = 'auto_review_trust_promotion'
TRUSTED_FINGERPRINT_PROJECTION_COMPONENT = 'trusted_knowledge_fingerprints'
TRUSTED_FINGERPRINT_PROJECTION_SCHEMA = 'trusted-fingerprint-projection:v1'
KEY_MATERIAL_VERIFIER_MARKER = b'paraworks:auto-review-key-material-verifier:v1'


class AutoReviewKeyBootstrapError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class AutoReviewKeyBootstrapResult:
    schema_available: bool
    initialized: bool
    ready: bool


def fingerprint_key_material_verifier(secret: str) -> str:
    return hmac.new(
        secret.encode('utf-8'),
        KEY_MATERIAL_VERIFIER_MARKER,
        hashlib.sha256,
    ).hexdigest()


class AutoReviewKeyBootstrapService:
    def __init__(
        self,
        *,
        session_factory: sessionmaker,
        settings: Settings,
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings

    def ensure_initialized(self) -> AutoReviewKeyBootstrapResult:
        with self._session_factory() as db:
            if not _c5_schema_available(db):
                db.rollback()
                return AutoReviewKeyBootstrapResult(
                    schema_available=False,
                    initialized=False,
                    ready=False,
                )

            dialect = db.get_bind().dialect.name
            sqlite_disabled = (
                dialect == 'sqlite' and self._settings.auto_review_mode == 'disabled'
            )
            if not sqlite_disabled:
                try:
                    self._settings.require_c5_durable_key_ready()
                except ValueError as exc:
                    raise AutoReviewKeyBootstrapError(
                        'durable_key_not_ready', str(exc)
                    ) from exc

            key_version = self._settings.agent_runtime_fingerprint_key_version.strip()
            verifier = fingerprint_key_material_verifier(
                self._settings.agent_runtime_fingerprint_secret
            )
            with KeyedMutationGuard.generation_barrier(db, exclusive=True):
                runtime = KeyedMutationGuard.lock_runtime_key_state(
                    db, for_update=True
                )
                retained = _has_retained_c5_state(db, ignore_runtime=True)
                if runtime is None and retained:
                    raise AutoReviewKeyBootstrapError(
                        'retained_keyed_state_without_runtime_identity',
                        'retained C.5 keyed state exists without runtime key identity',
                    )
                if runtime is not None:
                    _require_runtime_identity(
                        runtime,
                        key_version=key_version,
                        verifier=verifier,
                    )
                    projection = db.scalar(
                        select(TrustedKnowledgeFingerprintProjectionState).where(
                            TrustedKnowledgeFingerprintProjectionState.component
                            == TRUSTED_FINGERPRINT_PROJECTION_COMPONENT
                        )
                    )
                    if projection is None:
                        raise AutoReviewKeyBootstrapError(
                            'projection_state_missing',
                            'runtime key identity exists without projection state',
                        )
                    _require_projection_identity(projection, runtime=runtime)
                    db.commit()
                    return AutoReviewKeyBootstrapResult(
                        schema_available=True,
                        initialized=False,
                        ready=bool(runtime.ready),
                    )

                runtime = AutoReviewRuntimeKeyState(
                    component=AUTO_REVIEW_KEY_COMPONENT,
                    fingerprint_key_version=key_version,
                    fingerprint_key_material_verifier=verifier,
                    generation=1,
                    ready=dialect == 'postgresql',
                )
                projection = TrustedKnowledgeFingerprintProjectionState(
                    component=TRUSTED_FINGERPRINT_PROJECTION_COMPONENT,
                    projection_schema_version=(
                        TRUSTED_FINGERPRINT_PROJECTION_SCHEMA
                    ),
                    fingerprint_key_version=key_version,
                    fingerprint_key_material_verifier=verifier,
                    generation=1,
                    ready=False,
                    source_active_count=0,
                    projected_active_count=0,
                    rebuild_required=True,
                )
                db.add_all([runtime, projection])
                db.commit()
                return AutoReviewKeyBootstrapResult(
                    schema_available=True,
                    initialized=True,
                    ready=bool(runtime.ready),
                )


def _c5_schema_available(db: Session) -> bool:
    tables = set(inspect(db.get_bind()).get_table_names())
    return {
        'auto_review_runtime_key_states',
        'trusted_knowledge_fingerprint_projection_states',
    } <= tables


def _require_runtime_identity(
    runtime: AutoReviewRuntimeKeyState,
    *,
    key_version: str,
    verifier: str,
) -> None:
    if (
        runtime.fingerprint_key_version != key_version
        or runtime.fingerprint_key_material_verifier != verifier
    ):
        raise AutoReviewKeyBootstrapError(
            'runtime_key_identity_mismatch',
            'configured key identity does not match retained runtime state',
        )


def _require_projection_identity(
    projection: TrustedKnowledgeFingerprintProjectionState,
    *,
    runtime: AutoReviewRuntimeKeyState,
) -> None:
    if (
        projection.fingerprint_key_version != runtime.fingerprint_key_version
        or projection.fingerprint_key_material_verifier
        != runtime.fingerprint_key_material_verifier
        or projection.generation != runtime.generation
    ):
        raise AutoReviewKeyBootstrapError(
            'projection_key_identity_mismatch',
            'projection key identity does not match retained runtime state',
        )


NEW_C5_TABLES = (
    'auto_review_provider_safety_events',
    'auto_review_provider_safety_states',
    'trusted_knowledge_fingerprint_projection_states',
    'trusted_knowledge_fingerprints',
    'review_item_evidence_refs',
    'auto_review_extraction_calls',
    'auto_review_validation_calls',
    'auto_review_validations',
    'trusted_knowledge_approval_links',
    'trusted_knowledge_evidence_links',
    'assistant_message_evidence_dependencies',
    'assistant_message_knowledge_evidence_refs',
    'auto_review_rollout_states',
    'auto_review_rollout_control_events',
    'auto_review_promotion_decisions',
    'auto_review_post_audits',
    'auto_review_revocation_assessments',
    'auto_review_audit_corrections',
    'vector_serving_tombstones',
)


def has_retained_c5_state(db: Session) -> bool:
    return _has_retained_c5_state(db, ignore_runtime=False)


def _has_retained_c5_state(db: Session, *, ignore_runtime: bool) -> bool:
    tables = set(inspect(db.get_bind()).get_table_names())
    inspected_tables = list(NEW_C5_TABLES)
    if not ignore_runtime:
        inspected_tables.append('auto_review_runtime_key_states')
    for table_name in inspected_tables:
        if table_name in tables and db.execute(
            text(f'SELECT 1 FROM {table_name} LIMIT 1')
        ).first():
            return True

    predicates = {
        'agent_workflow_threads': "graph_version = 'company-memory-review-v2.1-auto-review'",
        'agent_workflow_requests': (
            'auto_review_mode IS NOT NULL OR '
            'fingerprint_key_material_verifier IS NOT NULL OR '
            'confirmed_total_cost_ceiling_usd IS NOT NULL'
        ),
        'review_items': (
            "candidate_contract_version IS NOT NULL OR agent_run_id IS NOT NULL OR "
            "resolution_source IS NOT NULL OR auto_validation_id IS NOT NULL OR "
            "revoked_at IS NOT NULL OR status = 'revoked'"
        ),
        'agent_runs': (
            'generation_provider IS NOT NULL OR '
            'generation_reasoning_effort IS NOT NULL OR '
            'generation_route_version IS NOT NULL OR '
            'generation_output_contract_version IS NOT NULL'
        ),
        'assistant_messages': (
            'evidence_contract_version IS NOT NULL OR '
            'serving_dependency_count IS NOT NULL'
        ),
        'sources': (
            'server_content_signature_schema IS NOT NULL OR '
            'server_content_signature IS NOT NULL OR '
            'connector_content_signature IS NOT NULL'
        ),
        'documents': 'current_document_version_id IS NOT NULL',
        'document_parser_runs': (
            'server_content_signature_schema IS NOT NULL OR '
            'server_content_signature IS NOT NULL OR '
            'parser_policy_version IS NOT NULL OR parser_version IS NOT NULL OR '
            'chunk_policy_version IS NOT NULL'
        ),
        'document_chunks': 'parser_run_id IS NOT NULL',
    }
    for table_name, predicate in predicates.items():
        if table_name not in tables:
            continue
        columns = {item['name'] for item in inspect(db.get_bind()).get_columns(table_name)}
        mentioned = {
            token
            for token in predicate.replace('(', ' ').replace(')', ' ').split()
            if token.isidentifier()
        }
        if not (mentioned & columns):
            continue
        try:
            retained = db.execute(
                text(f'SELECT 1 FROM {table_name} WHERE {predicate} LIMIT 1')
            ).first()
        except Exception:
            db.rollback()
            continue
        if retained:
            return True
    return False
