"""One fail-closed projection boundary for stored Assistant answers.

Integrity authenticates stored bytes; only the current canonical resolver grants
serving eligibility. Historical evidence never enters the V2 resolver by inference.
"""

import math
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from hmac import compare_digest
from types import MappingProxyType
from typing import get_args

from sqlalchemy import inspect, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.agent_runtime.provider_usage import AssistantSafePersistedErrorOutcome
from backend.app.agent_runtime.rag_v2_identity import (
    ServerRagSecurityScopeResolver,
    exact_utf8_bytes,
)
from backend.app.assistant.evidence_persistence import _dependency_storage_values
from backend.app.core.config import Settings, get_settings
from backend.app.core.demo_auth import DemoUser
from backend.app.models import (
    AgentRun,
    AssistantConversation,
    AssistantMessage,
    AssistantMessageEvidenceDependency,
    AssistantMessageKnowledgeEvidenceRef,
    AutoReviewRuntimeKeyState,
)
from backend.app.rag.evidence_projection import (
    ModelInfluenceDependencySnapshot,
    PreparedModelInfluenceObservation,
    build_model_influence_dependency_hmac,
    build_model_influence_set_hmac,
    build_v1_citation_projection_hmac,
    build_v1_selected_evidence_projection_hmac,
)
from backend.app.rag.serving_contracts import strictest_permission
from backend.app.rag.source_observations import CanonicalSourceObservationResolver
from backend.app.rag.trusted_evidence import (
    ServingEvidenceResolver,
    TrustedServingEnvelopeResolver,
)
from backend.app.schemas.rag import RagCitationResponse

UNAVAILABLE_CONTENT = '이 답변의 근거를 더 이상 확인할 수 없습니다. 다시 생성해 주세요.'
_CANNED_IDENTITIES = {
    **dict.fromkeys(
        get_args(AssistantSafePersistedErrorOutcome), 'rag-canned-generation-failure:v1'
    ),
    'no_match': 'rag-canned-no-evidence:v1',
    'hidden_only': 'rag-canned-no-evidence:v1',
    'safety_filter_empty': 'rag-canned-no-evidence:v1',
    'insufficient_evidence': 'rag-canned-no-evidence:v1',
    'evidence_unavailable': 'rag-canned-evidence-unavailable:v1',
    'budget_exceeded': 'rag-canned-budget-failure:v1',
}
_PUBLIC_RAG_METADATA = {
    'agent_name',
    'prompt_version',
    'status',
    'regeneration_required',
    'failure_reason',
    'failure_class',
}
_PRIVATE_METADATA = {
    'question',
    'graph_version',
    'effective_backend',
    'configured_backend',
    'fallback_category',
    'exception',
    'error',
    'error_message',
    'raw_exception',
    'parent_agent_run_id',
    'agent_run_id',
    'linked_agent_run_id',
    'run_id',
    'rag_result_hmac',
    'model_influence_set_hmac',
    'runtime_cost_snapshot_hmac',
    'retrieval_query_text',
    'raw_prompt',
    'model_response',
    'provider_trace',
}


@dataclass(frozen=True, slots=True)
class AssistantMessageView:
    id: int
    conversation_id: int
    role: str
    content: str
    citations: tuple[RagCitationResponse, ...]
    source_ids: tuple[str, ...]
    source_links: tuple[str, ...]
    source_snippets: tuple[str, ...]
    permission_level: str | None
    hidden_match_count: int
    permission_notice: str | None
    agent_run_id: int | None
    metadata: Mapping[str, object]
    created_at: str
    evidence_available: bool
    content_write_mode: str | None
    actor_id: str | None

    @property
    def metadata_(self) -> dict:
        """A detached copy for legacy email consumers, never mutable verified state."""
        return _thaw(self.metadata)

    def to_response(self) -> dict:
        return {
            name: (
                [
                    item.model_dump(mode='json', exclude_unset=True)
                    for item in self.citations
                ]
                if name == 'citations'
                else list(value)
                if name in {'source_ids', 'source_links', 'source_snippets'}
                else _thaw(value)
                if name == 'metadata'
                else value
            )
            for name in self.__slots__
            if name not in {'evidence_available', 'content_write_mode', 'actor_id'}
            for value in (getattr(self, name),)
        }


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def public_metadata(message: AssistantMessage) -> dict:
    metadata = message.metadata_ or {}
    if message.content_write_mode == 'rag_v2_exact' or (
        message.role == 'assistant'
        and 'action_type' not in metadata
        and (
            metadata.get('agent_name') == 'rag_orchestrator_agent'
            or metadata.get('prompt_version') in {'rag-answer:v1', 'rag-answer:v2'}
        )
    ):
        return {
            key: value
            for key, value in metadata.items()
            if key in _PUBLIC_RAG_METADATA
            and (
                (key == 'regeneration_required' and type(value) is bool)
                or (
                    key != 'regeneration_required'
                    and type(value) is str
                    and re.fullmatch(r'[A-Za-z0-9_:.-]{1,80}', value) is not None
                )
            )
        }

    # Non-RAG email/contact UI metadata is part of V1. Remove diagnostic fields
    # recursively, retaining the legacy failure classification (never error text).
    def clean(value):
        if isinstance(value, dict):
            return {
                key: clean(item)
                for key, item in value.items()
                if key not in _PRIVATE_METADATA
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return clean(metadata)


class AssistantEvidenceReader:
    def __init__(self, *, settings: Settings | None = None) -> None:
        self._settings = settings

    def project_message(
        self,
        *,
        db: Session | None,
        actor: DemoUser | None,
        message: AssistantMessage | AssistantMessageView,
    ) -> AssistantMessageView:
        if isinstance(message, AssistantMessageView):
            if message.actor_id == (actor.id if actor else None):
                return message
            return self._unavailable(message.id, message.conversation_id)

        # Identity/state access cannot trigger an expired attribute refresh. Keep
        # this fallback independent of the row surviving the fresh read below.
        state = inspect(message)
        message_id = state.identity[0] if state.identity else state.dict.get('id', 0)
        conversation_id = state.dict.get('conversation_id', 0)
        envelope = dict(state.dict)
        try:
            if db is not None:
                # Reject pending signature/provenance edits without flushing them.
                # Current source authority is read independently, never from the
                # caller identity map (even when an unrelated field is dirty).
                if (
                    message in db.dirty
                    and db.is_modified(message, include_collections=True)
                    or message in db.deleted
                ):
                    return self._unavailable(
                        message_id, conversation_id, envelope=envelope
                    )
                # Join the existing connection without transaction ownership.
                # Closing this read session neither commits nor rolls back the
                # caller transaction, and its fresh identity map cannot expire
                # caller rows or conceal committed revocation behind dirty state.
                with (
                    db.no_autoflush,
                    Session(
                        bind=db.connection(),
                        autoflush=False,
                        join_transaction_mode='rollback_only',
                    ) as reader_db,
                ):
                    fresh = reader_db.get(AssistantMessage, message_id)
                    if fresh is None:
                        return self._unavailable(
                            message_id, conversation_id, envelope=envelope
                        )
                    if self._has_pending_integrity_changes(db, fresh):
                        return self._unavailable(
                            fresh.id,
                            fresh.conversation_id,
                            envelope=inspect(fresh).dict,
                        )
                    return self._project_loaded(reader_db, actor, fresh)
            return self._project_loaded(db, actor, message)
        except (SQLAlchemyError, TypeError, ValueError, KeyError, AttributeError):
            return self._unavailable(message_id, conversation_id, envelope=envelope)

    @staticmethod
    def _has_pending_integrity_changes(caller_db, message):
        if message.role == 'user':
            return False  # Ordinary input has no Assistant integrity dependencies.
        for row in set(caller_db.dirty) | set(caller_db.deleted):
            state = inspect(row)
            if isinstance(row, AgentRun):
                relevant = bool(
                    state.identity and state.identity[0] == message.agent_run_id
                )
            elif isinstance(
                row,
                (
                    AssistantMessageEvidenceDependency,
                    AssistantMessageKnowledgeEvidenceRef,
                ),
            ):
                relevant = (
                    state.dict.get('assistant_message_id', message.id) == message.id
                    or state.committed_state.get('assistant_message_id') == message.id
                )
            elif isinstance(row, AutoReviewRuntimeKeyState):
                relevant = message.content_write_mode is not None
            else:
                relevant = False
            if relevant and (
                row in caller_db.deleted
                or caller_db.is_modified(row, include_collections=True)
            ):
                return True
        return False

    @staticmethod
    def _unavailable(message_id, conversation_id, *, envelope=None):
        envelope = envelope or {}
        created_at = envelope.get('created_at')
        role = envelope.get('role')
        return AssistantMessageView(
            id=message_id,
            conversation_id=conversation_id,
            role=role
            if type(role) is str and role in {'user', 'assistant'}
            else 'assistant',
            content=UNAVAILABLE_CONTENT,
            citations=(),
            source_ids=(),
            source_links=(),
            source_snippets=(),
            permission_level=None,
            hidden_match_count=0,
            permission_notice='evidence_unavailable',
            agent_run_id=envelope.get('agent_run_id')
            if type(envelope.get('agent_run_id')) is int
            else None,
            metadata=_freeze(
                {'status': 'evidence_unavailable', 'regeneration_required': True}
            ),
            created_at=created_at.isoformat()
            if isinstance(created_at, datetime)
            else '1970-01-01T00:00:00',
            evidence_available=False,
            content_write_mode=None,
            actor_id=None,
        )

    def _project_loaded(self, db, actor, message):
        if not self._is_live(db=db, actor=actor, message=message):
            return self._unavailable(
                message.id, message.conversation_id, envelope=inspect(message).dict
            )
        citation_reader = (
            RagCitationResponse.model_validate
            if message.content_write_mode == 'rag_v2_exact'
            else lambda item: RagCitationResponse.model_construct(
                **{
                    key: tuple(value) if key == 'matched_terms' else value
                    for key, value in item.items()
                }
            )
        )
        return AssistantMessageView(
            id=message.id,
            conversation_id=message.conversation_id,
            role=message.role,
            content=message.content,
            citations=tuple(citation_reader(item) for item in message.citations),
            source_ids=tuple(message.source_ids),
            source_links=tuple(message.source_links),
            source_snippets=tuple(message.source_snippets),
            permission_level=message.permission_level,
            hidden_match_count=message.hidden_match_count,
            permission_notice=message.permission_notice,
            agent_run_id=message.agent_run_id,
            metadata=_freeze(public_metadata(message)),
            created_at=message.created_at.isoformat(),
            evidence_available=True,
            content_write_mode=message.content_write_mode,
            actor_id=actor.id if actor else None,
        )

    def _is_live(self, *, db, actor, message):
        from backend.app.assistant.service import _message_evidence_is_live

        if message.role not in {'user', 'assistant'}:
            return False
        if message.role == 'user' and message.content_write_mode is not None:
            return False

        if db is not None and actor is not None:
            owner = db.scalar(
                select(AssistantConversation.user_id).where(
                    AssistantConversation.id == message.conversation_id
                )
            )
            if owner != actor.id:
                return False

        if message.content_write_mode is None:
            if any(
                getattr(message, field) is not None
                for field in (
                    'content_hmac_schema_version',
                    'assistant_message_content_hmac',
                    'content_hmac_key_version',
                    'content_hmac_key_material_verifier',
                    'content_origin',
                    'content_origin_hmac',
                    'rag_result_hmac',
                    'linked_agent_run_id',
                    'dependency_set_hmac_schema_version',
                    'dependency_set_hmac',
                    'parent_selected_evidence_projection_hmac',
                    'model_influence_set_hmac',
                )
            ):
                return False
            if message.role == 'user':
                # User-authored input is not an evidence-derived Assistant answer.
                # Do not run serving eligibility on it, but reject evidence marks.
                return not bool(
                    message.citations
                    or message.source_ids
                    or message.source_links
                    or message.source_snippets
                    or message.serving_dependency_count
                    or message.evidence_contract_version not in {None, 'none-v1'}
                    or (message.metadata_ or {}).get('evidence_derived')
                )
            if db is None or actor is None:
                return not bool(
                    message.citations
                    or message.source_ids
                    or message.source_links
                    or message.source_snippets
                    or (message.metadata_ or {}).get('evidence_derived')
                )
            return _message_evidence_is_live(db, user=actor, message=message)
        if message.content_write_mode != 'rag_v2_exact' or db is None or actor is None:
            return bool(
                message.content_write_mode == 'legacy_trimmed'
                and db is not None
                and actor is not None
                and self._legacy_is_live(db, actor, message)
            )
        return self._v2_is_live(db, actor, message)

    def _legacy_is_live(self, db, actor, message):
        """Read signed V1 snapshots without constructing D serving provenance.

        The legacy snapshot's version fingerprint is the existing V1 serving
        content hash; the two D identity/version child columns remain NULL.
        """
        from backend.app.assistant.service import _serving_content_hash
        from backend.app.knowledge.serving_text import canonical_knowledge_text
        from backend.app.knowledge.trusted_serving_eligibility import (
            TrustedServingEligibilityService,
            knowledge_model_for_type,
        )
        from backend.app.rag.serving_contracts import (
            build_canonical_citation_projection_hmac,
            build_model_content_hmac,
        )

        settings = self._settings or get_settings()
        secret, version = fingerprint_secret_bytes(settings)
        verifier = fingerprint_key_material_verifier(secret.decode('utf-8'))

        def fp(payload, schema, policy='assistant-evidence:v1'):
            return keyed_fingerprint(
                payload, secret=secret, schema_version=schema, policy_version=policy
            )

        if (
            message.role != 'assistant'
            or message.content_origin != 'legacy_evidence'
            or message.content_hmac_schema_version
            != 'assistant-message-content-hmac:v1'
            or message.dependency_set_hmac_schema_version
            != 'assistant-dependency-set-hmac:v2'
            or message.evidence_contract_version != 'assistant-evidence:v1'
            or message.content_hmac_key_version != version
            or message.content_hmac_key_material_verifier != verifier
            or message.linked_agent_run_id is not None
            or message.rag_result_hmac is not None
            or message.model_influence_set_hmac is not None
            or message.content != message.content.strip()
        ):
            return False
        runtime = db.scalar(
            select(AutoReviewRuntimeKeyState).where(
                AutoReviewRuntimeKeyState.component == 'auto_review_trust_promotion'
            )
        )
        if runtime is None or (
            not runtime.ready
            or runtime.fingerprint_key_version != version
            or runtime.fingerprint_key_material_verifier != verifier
        ):
            return False
        children = list(
            db.scalars(
                select(AssistantMessageEvidenceDependency)
                .where(
                    AssistantMessageEvidenceDependency.assistant_message_id
                    == message.id
                )
                .order_by(AssistantMessageEvidenceDependency.candidate_ordinal)
            )
        )
        if (
            not children
            or len(children) != message.serving_dependency_count
            or len(children) != len(message.citations)
        ):
            return False
        citation_hmacs = []
        for ordinal, child in enumerate(children):
            if (
                child.candidate_ordinal != ordinal
                or child.dependency_kind != 'legacy_unbound'
                or child.dependency_serving_scope != 'legacy_v1_only'
                or child.dependency_role != 'selected_citation'
                or child.fingerprint_key_version != version
                or child.fingerprint_key_material_verifier != verifier
                or child.dependency_set_hmac != message.dependency_set_hmac
                or child.legacy_human_base
                or any(
                    getattr(child, field) is not None
                    for field in (
                        'document_chunk_id',
                        'document_version_id',
                        'source_id',
                        'parser_run_id',
                        'knowledge_type',
                        'knowledge_id',
                        'approval_link_id',
                        'approval_provenance_hmac',
                        'evidence_link_set_hmac',
                        'serving_identity_hmac',
                        'serving_version_fingerprint',
                        'support_mode',
                        'legacy_source_review_item_id',
                        'current_document_version_id',
                        'server_content_signature_schema',
                        'server_content_signature',
                        'parser_policy_version',
                        'parser_version',
                        'chunk_policy_version',
                    )
                )
            ):
                return False
            kind, identifier = child.serving_document_id.split(':')
            if (
                kind
                not in {'decision_record', 'history_event', 'timeline_event', 'todo'}
                or str(int(identifier)) != identifier
                or int(identifier) <= 0
            ):
                return False
            target = db.get(knowledge_model_for_type(kind), int(identifier))
            current = TrustedServingEligibilityService(db).for_document(
                child.serving_document_id
            )
            if (
                target is None
                or not current.eligible
                or current.effective_permission != child.permission_level
                or child.permission_level not in actor.permission_levels
            ):
                return False
            if (
                db.scalar(
                    select(AssistantMessageKnowledgeEvidenceRef.id).where(
                        AssistantMessageKnowledgeEvidenceRef.dependency_id == child.id
                    )
                )
                is not None
            ):
                return False
            links, snippets = target.source_links, target.source_snippets
            if not links or len(links) != len(snippets):
                return False
            text = canonical_knowledge_text(kind, target)
            legacy_hash = _serving_content_hash(
                source_id=child.serving_document_id,
                text=text,
                source_url=links[0],
                source_snippet=snippets[0],
                permission_level=current.effective_permission,
            )
            model_hmac = build_model_content_hmac(
                serving_kind='trusted_knowledge', model_content=text, settings=settings
            )
            canonical_hmac = build_canonical_citation_projection_hmac(
                public_source_id=child.serving_document_id,
                public_source_type=kind,
                source_url=links[0],
                source_snippet=snippets[0],
                effective_permission=current.effective_permission,
                settings=settings,
            )
            identity = fp(
                {
                    'canonical_citation_projection_hmac': canonical_hmac,
                    'effective_permission': current.effective_permission,
                    'legacy_public_source_id_bytes': exact_utf8_bytes(
                        child.serving_document_id
                    ),
                    'legacy_source_links_bytes': [
                        exact_utf8_bytes(value) for value in links
                    ],
                    'legacy_source_snippets_bytes': [
                        exact_utf8_bytes(value) for value in snippets
                    ],
                    'model_content_hmac': model_hmac,
                    'serving_version_fingerprint': legacy_hash,
                },
                'assistant-legacy-dependency-snapshot:v1',
            )
            if (
                child.serving_content_hash != legacy_hash
                or child.model_content_hmac != model_hmac
                or child.canonical_citation_projection_hmac != canonical_hmac
                or child.legacy_dependency_identity_hmac != identity
            ):
                return False
            citation = message.citations[ordinal]
            canonical_fields = {
                'source_id': child.serving_document_id,
                'source_url': links[0],
                'source_type': kind,
                'permission_level': current.effective_permission,
                'source_snippet': snippets[0],
            }
            if any(
                citation.get(key) != value for key, value in canonical_fields.items()
            ):
                return False
            score, terms = citation['relevance_score'], citation['matched_terms']
            if (
                type(score) not in {float, int}
                or not math.isfinite(score)
                or type(terms) is not list
            ):
                return False
            citation_hmac = fp(
                {
                    'source_id_bytes': exact_utf8_bytes(child.serving_document_id),
                    'source_url_bytes': exact_utf8_bytes(links[0]),
                    'source_type_bytes': exact_utf8_bytes(kind),
                    'permission_level': current.effective_permission,
                    'source_snippet_bytes': exact_utf8_bytes(snippets[0]),
                    'relevance_score_binary64_be_hex': struct.pack('>d', score).hex(),
                    'matched_terms_bytes': [exact_utf8_bytes(term) for term in terms],
                },
                'rag-v1-evidence-projection:citation:v1',
                'rag-v1-evidence-projection:v1',
            )
            if child.selected_v1_citation_projection_hmac != citation_hmac:
                return False
            citation_hmacs.append(citation_hmac)
            child_hmac = fp(
                {
                    'approval_link_id': None,
                    'approval_provenance_hmac': None,
                    'candidate_ordinal': ordinal,
                    'canonical_citation_projection_hmac': canonical_hmac,
                    'dependency_kind': 'legacy_unbound',
                    'dependency_role': 'selected_citation',
                    'dependency_serving_scope': 'legacy_v1_only',
                    'effective_permission': child.permission_level,
                    'evidence_link_ids': [],
                    'evidence_link_set_hmac': None,
                    'legacy_dependency_identity_hmac': identity,
                    'model_content_hmac': model_hmac,
                    'raw_document_chunk_id': None,
                    'selected_v1_citation_projection_hmac': citation_hmac,
                    'serving_identity_hmac': None,
                    'serving_version_fingerprint': None,
                    'support_mode': None,
                    'trusted_knowledge_id': None,
                },
                'assistant-dependency-child-hmac:v2',
            )
            if child.dependency_child_hmac != child_hmac:
                return False
        selected_set = build_v1_selected_evidence_projection_hmac(
            citation_hmacs=tuple(citation_hmacs),
            source_ids=tuple(message.source_ids),
            source_links=tuple(message.source_links),
            source_snippets=tuple(message.source_snippets),
            settings=settings,
        )
        if selected_set != message.parent_selected_evidence_projection_hmac:
            return False
        metadata = message.metadata_ or {}
        agent, prompt = metadata.get('agent_name'), metadata.get('prompt_version')
        backend = metadata.get('effective_backend')
        if backend not in {'deterministic_lexical', 'pgvector'}:
            return False
        origin = fp(
            {
                'agent_name_bytes': exact_utf8_bytes(agent),
                'content_bytes_hmac': fp(
                    {'content_bytes': exact_utf8_bytes(message.content)},
                    'assistant-legacy-content-bytes:v1',
                ),
                'effective_backend': backend,
                'legacy_result_contract_version': 'rag-answer:v1',
                'output_permission': message.permission_level,
                'prompt_version_bytes': exact_utf8_bytes(prompt)
                if prompt is not None
                else None,
                'selected_evidence_projection_hmac': selected_set,
            },
            'assistant-legacy-evidence-origin:v1',
        )
        content = fp(
            {
                'agent_name_bytes': exact_utf8_bytes(agent),
                'assistant_message_id': message.id,
                'content_bytes': exact_utf8_bytes(message.content),
                'content_origin': 'legacy_evidence',
                'content_origin_hmac': origin,
                'content_write_mode': 'legacy_trimmed',
                'conversation_id': message.conversation_id,
                'linked_agent_run_id': None,
                'message_role': 'assistant',
                'prompt_version_bytes': exact_utf8_bytes(prompt)
                if prompt is not None
                else None,
                'rag_result_hmac': None,
            },
            'assistant-message-content-hmac:v1',
        )
        if (
            message.content_origin_hmac != origin
            or message.assistant_message_content_hmac != content
        ):
            return False
        whole_set = fp(
            {
                'assistant_message_content_hmac': content,
                'content_origin': 'legacy_evidence',
                'content_origin_hmac': origin,
                'content_write_mode': 'legacy_trimmed',
                'dependencies': [
                    {
                        'candidate_ordinal': child.candidate_ordinal,
                        'dependency_child_hmac': child.dependency_child_hmac,
                    }
                    for child in children
                ],
                'dependency_count': len(children),
                'dependency_serving_scope': 'legacy_v1_only',
                'evidence_contract_version': 'assistant-evidence:v1',
                'linked_agent_run_id': None,
                'model_influence_set_hmac': None,
                'parent_selected_evidence_projection_hmac': selected_set,
                'rag_result_hmac': None,
            },
            'assistant-dependency-set-hmac:v2',
        )
        return compare_digest(whole_set, message.dependency_set_hmac)

    def _v2_is_live(self, db, actor, message):
        from backend.app.assistant.service import (
            _assistant_message_has_valid_v2_structure,
            _canned_hidden_projection_is_valid,
            _knowledge_dependency_content_hash,
            _raw_dependency_content_hash,
        )

        if not _assistant_message_has_valid_v2_structure(db, message=message):
            return False
        settings = self._settings or get_settings()
        secret, version = fingerprint_secret_bytes(settings)
        verifier = fingerprint_key_material_verifier(secret.decode('utf-8'))
        if (
            message.content_hmac_key_version != version
            or message.content_hmac_key_material_verifier != verifier
        ):
            return False
        runtime = db.scalar(
            select(AutoReviewRuntimeKeyState).where(
                AutoReviewRuntimeKeyState.component == 'auto_review_trust_promotion'
            )
        )
        if runtime is None or (
            not runtime.ready
            or runtime.fingerprint_key_version != version
            or runtime.fingerprint_key_material_verifier != verifier
        ):
            return False
        parent = db.get(AgentRun, message.linked_agent_run_id)
        if not _canned_hidden_projection_is_valid(
            message
        ) or message.hidden_match_count != parent.metadata_.get(
            'hidden_match_count', 0
        ):
            return False
        expected = keyed_fingerprint(
            {
                'agent_name_bytes': exact_utf8_bytes(parent.agent_name),
                'assistant_message_id': message.id,
                'content_bytes': exact_utf8_bytes(message.content),
                'content_origin': message.content_origin,
                'content_origin_hmac': message.content_origin_hmac,
                'content_write_mode': message.content_write_mode,
                'conversation_id': message.conversation_id,
                'linked_agent_run_id': message.linked_agent_run_id,
                'message_role': message.role,
                'prompt_version_bytes': exact_utf8_bytes(parent.prompt_version),
                'rag_result_hmac': message.rag_result_hmac,
            },
            secret=secret,
            schema_version='assistant-message-content-hmac:v1',
            policy_version='assistant-evidence:v1',
        )
        if not compare_digest(expected, message.assistant_message_content_hmac):
            return False
        children = list(
            db.scalars(
                select(AssistantMessageEvidenceDependency)
                .where(
                    AssistantMessageEvidenceDependency.assistant_message_id
                    == message.id
                )
                .order_by(AssistantMessageEvidenceDependency.candidate_ordinal)
            )
        )
        if len(children) != message.serving_dependency_count:
            return False
        if message.content_origin == 'rag_canned':
            if children:
                return False
            identity = _CANNED_IDENTITIES.get(parent.metadata_.get('outcome'))
            if identity is None:
                return False
            return compare_digest(
                message.content_origin_hmac,
                keyed_fingerprint(
                    {'canned_message_identity': identity},
                    secret=secret,
                    schema_version='rag-canned-message-identity:v1',
                    policy_version='rag-answer:v2',
                ),
            )
        if (
            not 1 <= len(children) <= 8
            or parent.status != 'complete'
            or parent.metadata_.get('outcome') != 'supported'
            or message.dependency_set_hmac_schema_version
            != 'assistant-dependency-set-hmac:v2'
        ):
            return False
        scope = ServerRagSecurityScopeResolver(settings).resolve(db=db, actor=actor)
        snapshots = []
        selected_hmacs = []
        selected_index = 0
        for ordinal, child in enumerate(children):
            if (
                child.candidate_ordinal != ordinal
                or child.dependency_serving_scope != 'rag_v2'
                or child.dependency_role
                not in {'selected_citation', 'unselected_model_influence'}
                or child.fingerprint_key_version != version
                or child.fingerprint_key_material_verifier != verifier
                or child.dependency_set_hmac != message.dependency_set_hmac
                or child.legacy_dependency_identity_hmac is not None
            ):
                return False
            row = self._fresh_row(db, child, scope, settings)
            if row is None:
                return False
            values, link_ids = _dependency_storage_values(row.identity.version_envelope)
            if any(getattr(child, key) != value for key, value in values.items()):
                return False
            refs = list(
                db.scalars(
                    select(AssistantMessageKnowledgeEvidenceRef).where(
                        AssistantMessageKnowledgeEvidenceRef.dependency_id == child.id
                    )
                )
            )
            if any(
                ref.assistant_message_id != message.id
                or ref.approval_link_id != child.approval_link_id
                for ref in refs
            ) or sorted(ref.trusted_knowledge_evidence_link_id for ref in refs) != list(
                link_ids
            ):
                return False
            evidence = row.evidence
            if child.serving_document_id != evidence.serving_document_id:
                return False
            if child.dependency_kind == 'raw_chunk':
                from backend.app.models import DocumentChunk, Source

                legacy_content_hash = _raw_dependency_content_hash(
                    chunk=db.get(DocumentChunk, child.document_chunk_id),
                    source=db.get(Source, child.source_id),
                    permission_level=child.permission_level,
                )
            else:
                legacy_content_hash = _knowledge_dependency_content_hash(db, child)
            if child.serving_content_hash != legacy_content_hash:
                return False
            for name in (
                'model_content_hmac',
                'canonical_citation_projection_hmac',
                'serving_identity_hmac',
                'serving_version_fingerprint',
                'support_mode',
            ):
                if getattr(child, name) != getattr(evidence, name):
                    return False
            if (
                child.permission_level != evidence.effective_permission
                or child.approval_provenance_hmac != row.approval_provenance_hmac
                or child.evidence_link_set_hmac != row.evidence_link_set_hmac
            ):
                return False
            selected_hmac = None
            if child.dependency_role == 'selected_citation':
                if selected_index >= len(message.citations):
                    return False
                citation = message.citations[selected_index]
                if set(citation) != {
                    'source_id',
                    'source_url',
                    'source_type',
                    'permission_level',
                    'source_snippet',
                    'relevance_score',
                    'matched_terms',
                }:
                    return False
                for key, value in {
                    'source_id': evidence.public_source_id,
                    'source_url': row.source_url,
                    'source_type': evidence.public_source_type,
                    'permission_level': evidence.effective_permission,
                    'source_snippet': row.source_snippet,
                }.items():
                    if citation[key] != value:
                        return False
                selected_hmac = build_v1_citation_projection_hmac(
                    row=row,
                    relevance_score=citation['relevance_score'],
                    matched_terms=tuple(citation['matched_terms']),
                    settings=settings,
                )
                selected_hmacs.append(selected_hmac)
                selected_index += 1
            if selected_hmac != child.selected_v1_citation_projection_hmac:
                return False
            payload = {
                'approval_link_id': child.approval_link_id,
                'approval_provenance_hmac': child.approval_provenance_hmac,
                'candidate_ordinal': ordinal,
                'canonical_citation_projection_hmac': child.canonical_citation_projection_hmac,
                'dependency_kind': child.dependency_kind,
                'dependency_role': child.dependency_role,
                'dependency_serving_scope': child.dependency_serving_scope,
                'effective_permission': child.permission_level,
                'evidence_link_ids': list(link_ids),
                'evidence_link_set_hmac': child.evidence_link_set_hmac,
                'legacy_dependency_identity_hmac': None,
                'model_content_hmac': child.model_content_hmac,
                'raw_document_chunk_id': child.document_chunk_id,
                'selected_v1_citation_projection_hmac': selected_hmac,
                'serving_identity_hmac': child.serving_identity_hmac,
                'serving_version_fingerprint': child.serving_version_fingerprint,
                'support_mode': child.support_mode,
                'trusted_knowledge_id': child.knowledge_id,
            }
            expected_child = keyed_fingerprint(
                payload,
                secret=secret,
                schema_version='assistant-dependency-child-hmac:v2',
                policy_version='assistant-evidence:v1',
            )
            if not compare_digest(expected_child, child.dependency_child_hmac):
                return False
            observation = PreparedModelInfluenceObservation(
                ordinal,
                f'E{ordinal + 1}',
                evidence.support_mode,
                row.identity,
                evidence.effective_permission,
                evidence.serving_identity_hmac,
                evidence.serving_version_fingerprint,
                evidence.model_content_hmac,
                evidence.canonical_citation_projection_hmac,
                row.approval_provenance_hmac,
                row.evidence_link_set_hmac,
                '0' * 64,
            )
            snapshots.append(
                ModelInfluenceDependencySnapshot(
                    observation,
                    child.dependency_role,
                    row.identity,
                    build_model_influence_dependency_hmac(
                        row=row,
                        slot_id=observation.slot_id,
                        support_mode=evidence.support_mode,
                        dependency_role=child.dependency_role,
                        settings=settings,
                    ),
                )
            )
        if not selected_hmacs or selected_index != len(message.citations):
            return False
        selected_set = build_v1_selected_evidence_projection_hmac(
            citation_hmacs=tuple(selected_hmacs),
            source_ids=tuple(message.source_ids),
            source_links=tuple(message.source_links),
            source_snippets=tuple(message.source_snippets),
            settings=settings,
        )
        influence_set = build_model_influence_set_hmac(
            dependencies=tuple(snapshots), settings=settings
        )
        if (
            selected_set != message.parent_selected_evidence_projection_hmac
            or influence_set != message.model_influence_set_hmac
            or message.permission_level
            != strictest_permission(tuple(child.permission_level for child in children))
        ):
            return False
        expected_set = keyed_fingerprint(
            {
                'assistant_message_content_hmac': expected,
                'content_origin': message.content_origin,
                'content_origin_hmac': message.content_origin_hmac,
                'content_write_mode': message.content_write_mode,
                'dependencies': [
                    {
                        'candidate_ordinal': child.candidate_ordinal,
                        'dependency_child_hmac': child.dependency_child_hmac,
                    }
                    for child in children
                ],
                'dependency_count': len(children),
                'dependency_serving_scope': 'rag_v2',
                'evidence_contract_version': message.evidence_contract_version,
                'linked_agent_run_id': parent.id,
                'model_influence_set_hmac': influence_set,
                'parent_selected_evidence_projection_hmac': selected_set,
                'rag_result_hmac': message.rag_result_hmac,
            },
            secret=secret,
            schema_version='assistant-dependency-set-hmac:v2',
            policy_version='assistant-evidence:v1',
        )
        return compare_digest(expected_set, message.dependency_set_hmac)

    @staticmethod
    def _fresh_row(db, child, scope, settings):
        if child.dependency_kind == 'raw_chunk':
            return CanonicalSourceObservationResolver(
                db=db, settings=settings
            ).resolve_projection_for_scope_strict(child.document_chunk_id, scope=scope)
        if child.dependency_kind != 'trusted_knowledge':
            return None
        current = TrustedServingEnvelopeResolver(
            db=db, settings=settings
        ).resolve_for_scope_strict(
            child.knowledge_type, child.knowledge_id, scope=scope
        )
        if current is None:
            return None
        return ServingEvidenceResolver(
            settings=settings
        ).resolve_projection_candidate_strict(
            db=db, identity=current.identity, scope=scope
        )
