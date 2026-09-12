from __future__ import annotations

import os
import stat as stat_module
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from threading import RLock

from sqlalchemy import Engine, event, func, select
from sqlalchemy.orm import Session

from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.agent_runtime.keyed_mutation_guard import sqlite_keyed_mutation_mutex
from backend.app.agent_runtime.model_router import (
    build_rag_answer_model_config_snapshot_hmac,
)
from backend.app.agent_runtime.rag_finalization import (
    AssistantFinalizationRecord,
    AssistantProjectionTarget,
    CanonicalRagProjection,
    PreparedRagFinalization,
    RagFinalProjection,
    RagProjectionPending,
    _apply_parent_final,
    _build_result_hmac,
    _canned_text,
    _empty_answer_projection,
    _empty_search_projection,
    _hidden_membership_hmac,
    _output_permission,
    _pre_projection_cost_snapshot_hmac,
    _same_retrieval_authority,
    assistant_message_projection,
)
from backend.app.agent_runtime.rag_runtime_contracts import admission_source_window
from backend.app.agent_runtime.rag_safety_identity import admission_identity
from backend.app.agent_runtime.rag_v2_identity import (
    exact_utf8_bytes,
    security_scope_fingerprint,
)
from backend.app.agents.rag_orchestrator_agent.v2_answer_schema import (
    build_answer_output_schema_hmac,
    build_answer_prompt_renderer_hmac,
)
from backend.app.assistant.evidence_persistence import (
    AssistantEvidenceWriter,
    _mint_assistant_exact_write_authority,
)
from backend.app.core.config import Settings
from backend.app.models import (
    AgentRun,
    AgentRunCostComponent,
    AssistantConversation,
    AssistantMessage,
    AssistantMessageEvidenceDependency,
    RagServingCorpusGeneration,
)
from backend.app.rag.evidence_projection import (
    CanonicalEvidenceProjector,
    ProjectionFence,
    build_model_influence_set_hmac,
    build_v1_selected_evidence_projection_hmac,
)
from backend.app.rag.keyword_retriever import KeywordEvidenceRetriever
from backend.app.rag.retrieval import (
    RetrievalRequest,
    RetrievalResult,
    build_query_embedding_model_config_snapshot_hmac,
    rank_evidence_slots,
)
from backend.app.rag.search_store import SqlAlchemyKeywordSearchStore

_ZERO = Decimal('0.000000')
_SQLITE_RAG_SMOKE_MUTEX = sqlite_keyed_mutation_mutex()
_SQLITE_GRAPH_SCOPE_SEAL = object()


@dataclass(slots=True)
class _ProcessLock:
    handle: object
    database_identity: tuple[int, int]
    lock_identity: tuple[int, int]


_PROCESS_LOCKS: dict[Path, _ProcessLock] = {}


class SQLiteRagSmokeUnavailable(RuntimeError):  # noqa: N818 - approved API name
    pass


def sqlite_rag_smoke_mutex() -> RLock:
    """The single never-replaced process mutex shared by smoke writers."""
    return _SQLITE_RAG_SMOKE_MUTEX


class SQLiteRagSmokeCoordinator:
    """Own the provider-free SQLite retrieval and final product transaction."""

    def __init__(
        self,
        *,
        engine: Engine,
        database_path: Path | str | None,
        settings: Settings,
    ) -> None:
        if (
            not isinstance(engine, Engine)
            or engine.dialect.name != 'sqlite'
            or type(settings) is not Settings
            or settings.rag_retrieval_backend != 'keyword'
            or settings.rag_use_pgvector_search
            or settings.paraworks_env == 'production'
        ):
            raise SQLiteRagSmokeUnavailable(
                'SQLite RAG smoke requires keyword, provider-free, non-production mode'
            )
        self._engine = engine
        self._settings = settings
        self._secret, self._key_version = fingerprint_secret_bytes(settings)
        self._path = self._validated_path(database_path, settings)
        self._validate_engine_path()
        if self._path is not None:
            self._acquire_process_lock(self._path)

    @contextmanager
    def request_scope(self):
        """One SQLite authority for all graph operations; no interim commits."""
        with _SQLITE_RAG_SMOKE_MUTEX:
            self._revalidate_file_authority()
            with self._engine.connect() as connection:
                connection.exec_driver_sql('BEGIN IMMEDIATE')
                db = Session(bind=connection, join_transaction_mode='control_fully')
                db.begin()
                scope = SQLiteRagGraphScope(self, db, _seal=_SQLITE_GRAPH_SCOPE_SEAL)
                try:
                    yield scope
                    if scope.result is None:
                        raise SQLiteRagSmokeUnavailable('SQLite graph did not finalize a product')
                    self._assert_final_invariants(db, scope.result)
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise
                finally:
                    scope.active = False
                    db.close()

    def run_keyword(
        self,
        prepared: PreparedRagFinalization,
        *,
        assistant_target: AssistantProjectionTarget | None = None,
    ) -> RagFinalProjection:
        if (
            type(prepared) is PreparedRagFinalization
            and prepared.tentative_outcome == 'insufficient_evidence'
        ):
            raise SQLiteRagSmokeUnavailable(
                'insufficient_evidence is post-generation only'
            )
        if (
            type(prepared) is not PreparedRagFinalization
            or prepared.query_embedding_result is not None
            or prepared.retrieval_result.configured_backend != 'keyword'
            or (
                assistant_target is not None
                and type(assistant_target) is not AssistantProjectionTarget
            )
        ):
            raise SQLiteRagSmokeUnavailable('SQLite keyword smoke finalizer is unavailable')
        if (
            prepared.validated_answer is not None
            or prepared.prepared_model_influence is not None
            or prepared.model_influence_observations
            or prepared.selected_slot_ids
        ):
            raise SQLiteRagSmokeUnavailable(
                'SQLite substantive answer requires sealed internal fake-model authority'
            )
        with _SQLITE_RAG_SMOKE_MUTEX:
            self._revalidate_file_authority()
            connection = self._engine.connect()
            db: Session | None = None
            try:
                connection.exec_driver_sql('BEGIN IMMEDIATE')
                db = Session(bind=connection, join_transaction_mode='control_fully')
                result = self._finalize(db, prepared, assistant_target)
                self._assert_final_invariants(db, result)
                connection.commit()
                return result
            except Exception:
                connection.rollback()
                raise
            finally:
                if db is not None:
                    db.close()
                connection.close()

    def _finalize(
        self,
        db: Session,
        prepared: PreparedRagFinalization,
        assistant_target: AssistantProjectionTarget | None,
    ) -> RagFinalProjection:
        generation = db.get(RagServingCorpusGeneration, 1)
        if generation is None:
            raise SQLiteRagSmokeUnavailable('SQLite smoke serving generation is unavailable')
        scope_fingerprint = security_scope_fingerprint(
            prepared.security_scope, settings=self._settings
        )
        request = RetrievalRequest(
            retrieval_query_text=prepared.prepared_text.retrieval_query_text,
            security_scope=prepared.security_scope,
            security_scope_fingerprint=scope_fingerprint,
            query_embedding_result=None,
            candidate_scan_limit=50,
            visible_limit=5 if prepared.product_kind == 'search' else 8,
            relevance_policy_version='rag-retrieval-policy:v2.0',
        )
        fresh = KeywordEvidenceRetriever(
            store=SqlAlchemyKeywordSearchStore(db=db, settings=self._settings),
            settings=self._settings,
        ).invoke(request)
        drifted = not _same_retrieval_authority(fresh, prepared.retrieval_result)
        parent, children, pending = self._create_parent_and_costs(
            db,
            prepared=prepared,
            scope_fingerprint=scope_fingerprint,
            corpus_generation=generation.corpus_generation,
        )
        projection = self._project(
            db,
            pending=pending,
            prepared=prepared,
            fresh=fresh,
            current_corpus_generation=generation.corpus_generation,
            drifted=drifted,
        )
        _apply_parent_final(parent, projection, secret=self._secret)
        db.flush([parent, *children])
        if assistant_target is None:
            return projection
        return self._append_assistant(
            db,
            parent=parent,
            projection=projection,
            target=assistant_target,
            prepared=prepared,
        )

    def _create_parent_and_costs(
        self,
        db: Session,
        *,
        prepared: PreparedRagFinalization,
        scope_fingerprint: str,
        corpus_generation: int,
    ) -> tuple[AgentRun, tuple[AgentRunCostComponent, AgentRunCostComponent], RagProjectionPending]:
        surface = self._surface(prepared)
        return self._create_zero_cost_parent(db, surface=surface, prepared_text=prepared.prepared_text,
            scope_fingerprint=scope_fingerprint, corpus_generation=(prepared.prepared_model_influence.prepared_corpus_generation
                if prepared.prepared_model_influence is not None else corpus_generation))

    def _create_zero_cost_parent(self, db, *, surface, prepared_text, scope_fingerprint, corpus_generation):
        admission_hmac = admission_identity(
            {
                'answer_provider_policy_snapshot_hmac': None,
                'configured_backend': 'keyword',
                'current_text_hmac': prepared_text.current_text_hmac,
                'cutover_stage': surface,
                'graph_version': 'company-memory-rag-answer-v2.0',
                'mode': 'enforce',
                'query_context_version_bytes': exact_utf8_bytes(
                    prepared_text.query_context_version
                ),
                'query_embedding_provider_policy_snapshot_hmac': None,
                'retrieval_policy_version': 'rag-retrieval-policy:v2.0',
                'retrieval_query_hmac': prepared_text.retrieval_query_hmac,
                'security_scope_fingerprint': scope_fingerprint,
                'surface': surface,
            },
            secret=self._secret,
        )
        fence_hmac = keyed_fingerprint(
            {'admission_hmac': admission_hmac, 'corpus_generation': corpus_generation},
            secret=self._secret,
            schema_version='sqlite-rag-smoke-projection-fence:v1',
            policy_version='rag-run:v2',
        )
        parent = AgentRun(
            agent_name='rag_orchestrator_agent',
            prompt_version='rag-answer:v2',
            status='running',
            source_window=admission_source_window(
                mode='enforce', surface=surface, backend='keyword'
            ),
            cache_key='rag-v2-admission:' + admission_hmac,
            model_name='rag-v2-admission',
            generation_provider=None,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            estimated_cost_usd=0.0,
            permission_level='restricted',
            metadata_={},
            run_contract_version='rag-run:v2',
            run_record_phase='cost_finalized_pending_projection',
            total_charged_cost_usd=_ZERO,
            projection_owner_fence_hmac=fence_hmac,
            completed_at=None,
        )
        db.add(parent)
        db.flush([parent])
        children = (
            self._terminal_zero_child(parent.id, component='query_embedding'),
            self._terminal_zero_child(parent.id, component='answer_generation'),
        )
        db.add_all(children)
        db.flush(children)
        cost_hmac = _pre_projection_cost_snapshot_hmac(
            agent_run_id=parent.id, children=children, secret=self._secret
        )
        parent.metadata_ = {
            'configured_backend': 'keyword',
            'current_text_hmac': prepared_text.current_text_hmac,
            'query_context_version': prepared_text.query_context_version,
            'retrieval_query_hmac': prepared_text.retrieval_query_hmac,
            'runtime_cost_snapshot_hmac': cost_hmac,
            'security_scope_fingerprint': scope_fingerprint,
            'surface': surface,
        }
        pending = RagProjectionPending(
            parent_agent_run_id=parent.id,
            projection_owner_fence_hmac=fence_hmac,
            security_scope_fingerprint=scope_fingerprint,
            prepared_corpus_generation=corpus_generation,
            prepared_vector_index_generation=None,
            terminal_cost_snapshot_hmac=cost_hmac,
        )
        return parent, children, pending

    @staticmethod
    def _surface(prepared: PreparedRagFinalization) -> str:
        if prepared.product_kind == 'search':
            return 'search'
        if prepared.prepared_text.query_context_version == 'assistant-context:v1':
            return 'assistant'
        return 'ask'

    def _terminal_zero_child(
        self, parent_id: int, *, component: str
    ) -> AgentRunCostComponent:
        query = component == 'query_embedding'
        return AgentRunCostComponent(
            agent_run_id=parent_id,
            component=component,
            component_ordinal=0 if query else 1,
            dispatch_state='terminal',
            dispatch_fence_hmac=None,
            process_instance_hmac=None,
            attempted=False,
            dispatch_count=0,
            reserved_input_tokens=0,
            reserved_output_tokens=0,
            actual_input_tokens=None,
            actual_output_tokens=None,
            reserved_cost_usd=_ZERO,
            charged_cost_usd=_ZERO,
            charge_basis='zero',
            overrun=False,
            provider='openai',
            model=(self._settings.openai_embedding_model if query else 'gpt-5.4-mini-2026-03-17'),
            authorized_model_config_version=(
                'rag-query-embedding-config:v1' if query else 'rag-answer-model-config:v1'
            ),
            authorized_model_config_snapshot_hmac=(
                build_query_embedding_model_config_snapshot_hmac(self._settings)
                if query
                else build_rag_answer_model_config_snapshot_hmac(
                    self._settings,
                    output_schema_hmac=build_answer_output_schema_hmac(
                        self._settings
                    ),
                    prompt_renderer_hmac=build_answer_prompt_renderer_hmac(
                        self._settings
                    ),
                )
            ),
            authorized_cost_policy_version=(
                'rag-query-embedding-cost:v1' if query else 'rag-answer-cost:v1'
            ),
            authorized_token_estimator_version=(
                'openai-cl100k-text-embedding-3-small:v1'
                if query
                else 'openai-o200k-rag-answer:v1'
            ),
            authorized_policy_snapshot_hmac=keyed_fingerprint(
                {'component': component, 'provider_dispatch_count': 0},
                secret=self._secret,
                schema_version='sqlite-rag-smoke-provider-policy:v1',
                policy_version='rag-run:v2',
            ),
            terminal_outcome=None,
        )

    def _project(
        self,
        db: Session,
        *,
        pending: RagProjectionPending,
        prepared: PreparedRagFinalization,
        fresh: RetrievalResult,
        current_corpus_generation: int,
        drifted: bool,
    ) -> CanonicalRagProjection:
        fresh_slots = rank_evidence_slots(fresh.visible)
        fresh_hidden = _hidden_membership_hmac(fresh, secret=self._secret)
        prepared_hidden = _hidden_membership_hmac(
            prepared.retrieval_result, secret=self._secret
        )
        fence = ProjectionFence(
            prepared_corpus_generation=pending.prepared_corpus_generation,
            prepared_index_generation=None,
            current_corpus_generation=current_corpus_generation,
            current_index_generation=None,
            prepared_hidden_membership_hmac=prepared_hidden,
            current_hidden_membership_hmac=fresh_hidden,
        )
        projector = CanonicalEvidenceProjector(db=db, settings=self._settings)
        dependencies = ()
        if prepared.product_kind == 'search':
            if drifted:
                evidence = _empty_search_projection(self._settings)
                outcome = 'evidence_unavailable'
            else:
                evidence = projector.project_search(
                    fresh_slots[:5], scope=prepared.security_scope, fence=fence
                )
                outcome = 'search_projected'
            answer_text = None
        elif (
            not drifted
            and prepared.validated_answer is not None
            and prepared.validated_answer.blocks
            and prepared.prepared_model_influence is not None
        ):
            selected = prepared.validated_answer.selected_slot_ids
            evidence = projector.project_selected(
                fresh_slots,
                selected_slot_ids=selected,
                scope=prepared.security_scope,
                fence=fence,
            )
            dependencies = projector.finalize_model_influence_dependencies(
                prepared.prepared_model_influence,
                selected,
                scope=prepared.security_scope,
                fence=fence,
            )
            empty_hmac = build_v1_selected_evidence_projection_hmac(
                citation_hmacs=(),
                source_ids=(),
                source_links=(),
                source_snippets=(),
                settings=self._settings,
            )
            if dependencies and evidence.projection_hmac != empty_hmac:
                outcome = 'supported'
                answer_text = prepared.validated_answer.assembled_answer
            else:
                outcome = 'evidence_unavailable'
                answer_text = _canned_text('rag-canned-evidence-unavailable:v1')
                evidence = _empty_answer_projection(self._settings)
                dependencies = ()
        else:
            outcome = 'evidence_unavailable' if drifted else prepared.tentative_outcome
            answer_text = _canned_text(
                'rag-canned-evidence-unavailable:v1'
                if drifted
                else prepared.canned_message_identity
            )
            evidence = _empty_answer_projection(self._settings)
        result_hmac = _build_result_hmac(
            pending=pending,
            prepared=prepared,
            outcome=outcome,
            evidence=evidence,
            dependencies=dependencies,
            hidden_hmac=fresh_hidden,
            effective_backend=fresh.effective_backend,
            secret=self._secret,
            settings=self._settings,
        )
        return CanonicalRagProjection(
            outcome=outcome,
            answer_text=answer_text,
            evidence=evidence,
            model_influence=dependencies,
            hidden_match_count=(0 if outcome == 'evidence_unavailable' else fresh.hidden_match_count),
            effective_backend=fresh.effective_backend,
            result_hmac=result_hmac,
        )

    def _append_assistant(
        self,
        db: Session,
        *,
        parent: AgentRun,
        projection: CanonicalRagProjection,
        target: AssistantProjectionTarget,
        prepared: PreparedRagFinalization,
    ) -> AssistantFinalizationRecord:
        conversation = db.get(AssistantConversation, target.conversation_id)
        user_message = db.get(AssistantMessage, target.user_message_id)
        if (
            conversation is None
            or user_message is None
            or conversation.user_id != target.owner_user_id
            or user_message.conversation_id != conversation.id
            or user_message.role != 'user'
        ):
            raise SQLiteRagSmokeUnavailable('assistant projection target changed')
        canned = (
            None
            if projection.model_influence
            else (
                'rag-canned-evidence-unavailable:v1'
                if projection.outcome == 'evidence_unavailable'
                else 'rag-canned-no-evidence:v1'
            )
        )
        message_projection = assistant_message_projection(
            projection,
            metadata={'status': projection.outcome},
            canned_message_identity=canned,
            assembled_answer_hmac=(
                prepared.validated_answer.assembled_answer_hmac
                if projection.model_influence and prepared.validated_answer is not None
                else None
            ),
            permission_level=_output_permission(projection),
            permission_notice=(
                'evidence_unavailable' if projection.outcome == 'evidence_unavailable' else None
            ),
        )
        authority = _mint_assistant_exact_write_authority(
            parent_agent_run_id=parent.id,
            conversation_id=conversation.id,
            projection=message_projection,
            parent_result_hmac=parent.metadata_.get('rag_result_hmac'),
        )
        writer = AssistantEvidenceWriter(
            fingerprint_secret=self._secret,
            fingerprint_key_version=self._key_version,
            settings=self._settings,
        )
        reserved_message_id = (db.scalar(select(func.max(AssistantMessage.id))) or 0) + 1

        def seed_sqlite_exact_insert(session, flush_context, instances) -> None:
            del flush_context, instances
            for candidate in session.new:
                if (
                    type(candidate) is AssistantMessage
                    and candidate.linked_agent_run_id == parent.id
                    and candidate.assistant_message_content_hmac is None
                ):
                    candidate.id = reserved_message_id
                    candidate.assistant_message_content_hmac = keyed_fingerprint(
                        {
                            'agent_name_bytes': exact_utf8_bytes(
                                'rag_orchestrator_agent'
                            ),
                            'assistant_message_id': reserved_message_id,
                            'content_bytes': exact_utf8_bytes(candidate.content),
                            'content_origin': candidate.content_origin,
                            'content_origin_hmac': candidate.content_origin_hmac,
                            'content_write_mode': 'rag_v2_exact',
                            'conversation_id': conversation.id,
                            'linked_agent_run_id': parent.id,
                            'message_role': 'assistant',
                            'prompt_version_bytes': exact_utf8_bytes('rag-answer:v2'),
                            'rag_result_hmac': projection.result_hmac,
                        },
                        secret=self._secret,
                        schema_version='assistant-message-content-hmac:v1',
                        policy_version='assistant-evidence:v1',
                    )
                    if projection.model_influence:
                        candidate.model_influence_set_hmac = (
                            build_model_influence_set_hmac(
                                dependencies=projection.model_influence,
                                settings=self._settings,
                            )
                        )
                        candidate.dependency_set_hmac = '0' * 64

        event.listen(db, 'before_flush', seed_sqlite_exact_insert)
        try:
            message = writer.append_final(
                db=db,
                conversation=conversation,
                projection=message_projection,
                authority=authority,
            )
        finally:
            event.remove(db, 'before_flush', seed_sqlite_exact_insert)
        if projection.model_influence and message.dependency_set_hmac == '0' * 64:
            raise SQLiteRagSmokeUnavailable(
                'SQLite assistant dependency identity was not finalized'
            )
        return AssistantFinalizationRecord(
            assistant_message_id=message.id,
            parent_agent_run_id=parent.id,
            application_outcome=projection.outcome,
            finalization_kind='substantive' if projection.model_influence else 'canned_safe',
            canonical_projection=projection,
        )

    @staticmethod
    def _assert_final_invariants(db: Session, result: RagFinalProjection) -> None:
        parent_id = (
            result.parent_agent_run_id
            if type(result) is AssistantFinalizationRecord
            else db.scalar(select(AgentRun.id).order_by(AgentRun.id.desc()))
        )
        parent = db.get(AgentRun, parent_id)
        children = tuple(
            db.scalars(
                select(AgentRunCostComponent)
                .where(AgentRunCostComponent.agent_run_id == parent_id)
                .order_by(AgentRunCostComponent.component_ordinal)
            )
        )
        if (
            parent is None
            or parent.status != 'complete'
            or parent.run_contract_version != 'rag-run:v2'
            or parent.run_record_phase != 'final'
            or Decimal(parent.total_charged_cost_usd) != _ZERO
            or parent.projection_owner_fence_hmac is not None
            or tuple((row.component, row.component_ordinal) for row in children)
            != (('query_embedding', 0), ('answer_generation', 1))
            or any(
                row.dispatch_state != 'terminal'
                or row.attempted is not False
                or row.dispatch_count != 0
                or row.reserved_input_tokens != 0
                or row.reserved_output_tokens != 0
                or row.actual_input_tokens is not None
                or row.actual_output_tokens is not None
                or Decimal(row.reserved_cost_usd) != _ZERO
                or Decimal(row.charged_cost_usd) != _ZERO
                or row.charge_basis != 'zero'
                or row.overrun is not False
                or row.dispatch_fence_hmac is not None
                or row.process_instance_hmac is not None
                or row.terminal_outcome is not None
                for row in children
            )
        ):
            raise SQLiteRagSmokeUnavailable('SQLite smoke exact-two final invariant failed')
        if type(result) is AssistantFinalizationRecord:
            message = db.get(AssistantMessage, result.assistant_message_id)
            dependencies = tuple(
                db.scalars(
                    select(AssistantMessageEvidenceDependency)
                    .where(
                        AssistantMessageEvidenceDependency.assistant_message_id
                        == result.assistant_message_id
                    )
                    .order_by(
                        AssistantMessageEvidenceDependency.candidate_ordinal
                    )
                )
            )
            substantive = result.finalization_kind == 'substantive'
            if (
                message is None
                or message.role != 'assistant'
                or message.content_write_mode != 'rag_v2_exact'
                or message.linked_agent_run_id != parent_id
                or message.agent_run_id != parent_id
                or message.rag_result_hmac != parent.metadata_.get('rag_result_hmac')
                or message.metadata_.get('status') != result.application_outcome
                or parent.metadata_.get('outcome') != result.application_outcome
                or message.serving_dependency_count != len(dependencies)
                or substantive != bool(dependencies)
                or tuple(row.candidate_ordinal for row in dependencies)
                != tuple(range(len(dependencies)))
                or any(
                    row.assistant_message_id != message.id
                    or row.dependency_set_hmac != message.dependency_set_hmac
                    for row in dependencies
                )
            ):
                raise SQLiteRagSmokeUnavailable(
                    'SQLite smoke assistant linkage invariant failed'
                )
        elif parent.metadata_.get('rag_result_hmac') != result.result_hmac:
            raise SQLiteRagSmokeUnavailable(
                'SQLite smoke direct result linkage invariant failed'
            )
        pending = db.scalar(
            select(AgentRun.id)
            .where(
                AgentRun.run_contract_version == 'rag-run:v2',
                AgentRun.run_record_phase == 'cost_finalized_pending_projection',
            )
            .limit(1)
        )
        if pending is not None:
            raise SQLiteRagSmokeUnavailable('SQLite smoke cannot commit pending projection residue')

    def _validate_engine_path(self) -> None:
        configured = self._engine.url.database
        if self._path is None:
            if configured not in {None, '', ':memory:'}:
                raise SQLiteRagSmokeUnavailable('SQLite smoke engine/path identity mismatch')
            return
        if configured is None:
            raise SQLiteRagSmokeUnavailable('SQLite smoke engine/path identity mismatch')
        actual = Path(configured).absolute()
        try:
            if actual != self._path or actual.resolve(strict=True) != self._path:
                raise SQLiteRagSmokeUnavailable('SQLite smoke engine/path identity mismatch')
        except OSError as exc:
            raise SQLiteRagSmokeUnavailable('SQLite smoke engine identity is unavailable') from exc

    @staticmethod
    def _validated_path(
        database_path: Path | str | None,
        settings: Settings,
    ) -> Path | None:
        if database_path is None or str(database_path) == ':memory:':
            if settings.paraworks_env != 'test':
                raise SQLiteRagSmokeUnavailable('in-memory SQLite is automated-test-only')
            return None
        raw = Path(database_path)
        if not raw.is_absolute() or any(part in {'.', '..'} for part in raw.parts):
            raise SQLiteRagSmokeUnavailable('SQLite smoke database path identity is ambiguous')
        absolute = raw.absolute()
        try:
            if (
                raw.is_symlink()
                or str(absolute.resolve(strict=True)) != str(absolute)
            ):
                raise SQLiteRagSmokeUnavailable('SQLite smoke database path identity is ambiguous')
            SQLiteRagSmokeCoordinator._require_single_link_regular(absolute, label='database')
        except SQLiteRagSmokeUnavailable:
            raise
        except OSError as exc:
            raise SQLiteRagSmokeUnavailable('SQLite smoke database path is unavailable') from exc
        return absolute

    @staticmethod
    def _require_single_link_regular(path: Path, *, label: str) -> os.stat_result:
        value = path.lstat()
        if not stat_module.S_ISREG(value.st_mode):
            raise SQLiteRagSmokeUnavailable(f'SQLite smoke {label} must be a regular file')
        if getattr(value, 'st_file_attributes', 0) & 0x400:
            raise SQLiteRagSmokeUnavailable(
                f'SQLite smoke {label} reparse identity is unavailable'
            )
        if value.st_nlink != 1:
            raise SQLiteRagSmokeUnavailable(
                f'SQLite smoke {label} hardlink identity is unavailable'
            )
        return value

    def _revalidate_file_authority(self) -> None:
        if self._path is None:
            return
        lock_path = self._lock_path(self._path)
        record = _PROCESS_LOCKS.get(lock_path)
        if record is None:
            raise SQLiteRagSmokeUnavailable('SQLite smoke process lock authority is unavailable')
        database_stat = self._require_single_link_regular(self._path, label='database')
        lock_stat = self._require_single_link_regular(lock_path, label='lock')
        try:
            handle_stat = os.fstat(record.handle.fileno())
        except (AttributeError, OSError, ValueError) as exc:
            raise SQLiteRagSmokeUnavailable('SQLite smoke process lock authority is unavailable') from exc
        if (
            (database_stat.st_dev, database_stat.st_ino) != record.database_identity
            or (lock_stat.st_dev, lock_stat.st_ino) != record.lock_identity
            or (handle_stat.st_dev, handle_stat.st_ino) != record.lock_identity
        ):
            raise SQLiteRagSmokeUnavailable('SQLite smoke file authority identity changed')

    @staticmethod
    def _lock_path(database_path: Path) -> Path:
        return database_path.with_name(database_path.name + '.rag-smoke-process.lock')

    @classmethod
    def _acquire_process_lock(cls, database_path: Path) -> None:
        database_stat = cls._require_single_link_regular(database_path, label='database')
        lock_path = cls._lock_path(database_path)
        if lock_path in _PROCESS_LOCKS:
            record = _PROCESS_LOCKS[lock_path]
            lock_stat = cls._require_single_link_regular(lock_path, label='lock')
            try:
                handle_stat = os.fstat(record.handle.fileno())
            except (AttributeError, OSError, ValueError) as exc:
                raise SQLiteRagSmokeUnavailable(
                    'SQLite smoke process lock authority is unavailable'
                ) from exc
            if (
                (database_stat.st_dev, database_stat.st_ino)
                != record.database_identity
                or (lock_stat.st_dev, lock_stat.st_ino) != record.lock_identity
                or (handle_stat.st_dev, handle_stat.st_ino)
                != record.lock_identity
            ):
                raise SQLiteRagSmokeUnavailable(
                    'SQLite smoke file authority identity changed'
                )
            return
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, 'O_NOFOLLOW'):
            flags |= os.O_NOFOLLOW
        handle = None
        try:
            descriptor = os.open(lock_path, flags, 0o600)
            handle = os.fdopen(descriptor, 'a+b', buffering=0)
            lock_stat = cls._require_single_link_regular(lock_path, label='lock')
            handle_stat = os.fstat(handle.fileno())
            if (lock_stat.st_dev, lock_stat.st_ino) != (handle_stat.st_dev, handle_stat.st_ino):
                raise SQLiteRagSmokeUnavailable('SQLite smoke lock file identity mismatch')
            if os.name == 'nt':
                import msvcrt

                if lock_stat.st_size == 0:
                    handle.write(b'0')
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            _PROCESS_LOCKS[lock_path] = _ProcessLock(
                handle=handle,
                database_identity=(database_stat.st_dev, database_stat.st_ino),
                lock_identity=(lock_stat.st_dev, lock_stat.st_ino),
            )
        except SQLiteRagSmokeUnavailable:
            if handle is not None:
                handle.close()
            raise
        except (OSError, ValueError) as exc:
            if handle is not None:
                handle.close()
            raise SQLiteRagSmokeUnavailable('SQLite smoke process lock is unavailable') from exc


class SQLiteRagGraphScope:
    """Request-local operations under the coordinator's existing SQLite lock."""

    def __init__(self, coordinator: SQLiteRagSmokeCoordinator, db: Session, *, _seal):
        if _seal is not _SQLITE_GRAPH_SCOPE_SEAL:
            raise SQLiteRagSmokeUnavailable('SQLite graph scope is coordinator-owned')
        self.coordinator = coordinator
        self.db = db
        self.active = True
        self.parent = None
        self.children = ()
        self.pending = None
        self.result = None
        self.invocation = None
        self.answer = None
        self.answer_policy = None

    def _require_active(self):
        if not self.active or not self.db.in_transaction() or self.result is not None:
            raise SQLiteRagSmokeUnavailable('SQLite graph scope is unavailable')

    def admit(self, *, prepared_text, surface, security_scope):
        self._require_active()
        if self.parent is not None:
            raise SQLiteRagSmokeUnavailable('SQLite graph admission is single-use')
        generation = self.db.get(RagServingCorpusGeneration, 1)
        if generation is None:
            raise SQLiteRagSmokeUnavailable('SQLite smoke serving generation is unavailable')
        self.parent, self.children, self.pending = self.coordinator._create_zero_cost_parent(
            self.db, surface=surface, prepared_text=prepared_text,
            scope_fingerprint=security_scope_fingerprint(security_scope, settings=self.coordinator._settings),
            corpus_generation=generation.corpus_generation)
        return self.parent.id, (generation.corpus_generation, None)

    def mark_projection_pending(self, run_id):
        self._require_active()
        if self.parent is None or self.parent.id != run_id:
            raise SQLiteRagSmokeUnavailable('SQLite graph parent is unavailable')
        return self.pending

    def bind_answer_invocation(self, invocation, influence, *, model, cost_policy):
        self._require_active()
        if self.parent is None or self.invocation is not None:
            raise SQLiteRagSmokeUnavailable('SQLite answer preparation is single-use')
        model.validate_prepared_invocation(invocation)
        cost_policy.verify_answer_model_influence(invocation.evidence_slots, invocation.model_influence, prepared_set=influence)
        if invocation.retrieval_query_hmac != self.parent.metadata_['retrieval_query_hmac'] or invocation.rendered_input_hmac != influence.rendered_input_hmac:
            raise SQLiteRagSmokeUnavailable('SQLite answer identity changed')
        self.parent.metadata_ = {**self.parent.metadata_,
            'rendered_input_hmac': invocation.rendered_input_hmac,
            'answer_model_config_snapshot_hmac': invocation.model_config_snapshot_hmac,
            'prepared_model_influence_observation_hmac': influence.aggregate_observation_hmac}
        self.invocation = invocation
        self.answer_policy = cost_policy

    def generate_deterministic_answer(self, invocation):
        """Internal non-production fixture: no transport or external charge."""
        from backend.app.agents.rag_orchestrator_agent.v2_answer_schema import (
            RagAnswerOutputValidator,
        )
        self._require_active()
        if invocation is not self.invocation or self.answer is not None:
            raise SQLiteRagSmokeUnavailable('SQLite deterministic answer authority is unavailable')
        slot = invocation.evidence_slots[0]
        self.answer = RagAnswerOutputValidator(signer=self.answer_policy.sign_answer_artifact).validate({
            'answer_blocks': [{'text': slot.evidence.model_content[:350], 'evidence_slot_ids': [slot.slot_id], 'support_mode': slot.support_mode}],
            'insufficient_evidence_reason': None}, slots=invocation.evidence_slots)
        return self.answer

    def finalize(self, pending, prepared, *, assistant_target=None):
        self._require_active()
        if (pending is not self.pending or prepared.query_embedding_result is not None
            or prepared.validated_answer is not None and prepared.validated_answer is not self.answer):
            raise SQLiteRagSmokeUnavailable('SQLite graph finalization authority is unavailable')
        fresh = KeywordEvidenceRetriever(store=SqlAlchemyKeywordSearchStore(db=self.db, settings=self.coordinator._settings),
            settings=self.coordinator._settings).invoke(RetrievalRequest(
                retrieval_query_text=prepared.prepared_text.retrieval_query_text, security_scope=prepared.security_scope,
                security_scope_fingerprint=pending.security_scope_fingerprint, query_embedding_result=None,
                candidate_scan_limit=50, visible_limit=5 if prepared.product_kind == 'search' else 8,
                relevance_policy_version='rag-retrieval-policy:v2.0'))
        generation = self.db.get(RagServingCorpusGeneration, 1)
        projection = self.coordinator._project(self.db, pending=pending, prepared=prepared, fresh=fresh,
            current_corpus_generation=generation.corpus_generation,
            drifted=not _same_retrieval_authority(fresh, prepared.retrieval_result))
        _apply_parent_final(self.parent, projection, secret=self.coordinator._secret)
        self.db.flush([self.parent, *self.children])
        self.result = (projection if assistant_target is None else self.coordinator._append_assistant(
            self.db, parent=self.parent, projection=projection, target=assistant_target, prepared=prepared))
        return self.result
