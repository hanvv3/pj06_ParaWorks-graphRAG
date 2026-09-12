from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Literal

from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime.fingerprints import keyed_fingerprint
from backend.app.agent_runtime.provider_usage import RagCannedMessageIdentity
from backend.app.agent_runtime.rag_v2_identity import exact_utf8_bytes
from backend.app.core.config import Settings
from backend.app.models import (
    AssistantConversation,
    AssistantMessage,
    AssistantMessageEvidenceDependency,
    AssistantMessageKnowledgeEvidenceRef,
)
from backend.app.rag.evidence_projection import (
    ModelInfluenceDependencySnapshot,
    V1EvidenceProjection,
    build_model_influence_set_hmac,
    projection_record_to_transport,
)
from backend.app.rag.serving_contracts import (
    ExplicitApprovalProvenance,
    LegacyHumanProvenance,
    RawServingVersionEnvelope,
    TrustedServingVersionEnvelope,
)


@dataclass(frozen=True, slots=True)
class AssistantMessageProjection:
    content: str
    metadata: Mapping[str, object]
    evidence: V1EvidenceProjection
    permission_level: str | None
    permission_notice: str | None
    hidden_match_count: int
    assembled_answer_hmac: str | None
    canned_message_identity: RagCannedMessageIdentity | None
    result_hmac: str
    model_influence: tuple[ModelInfluenceDependencySnapshot, ...] = ()

    def __post_init__(self) -> None:
        if type(self.content) is not str or not self.content.strip():
            raise ValueError('V2 assistant content must be nonblank')
        if not isinstance(self.metadata, Mapping):
            raise ValueError('V2 assistant metadata is invalid')
        if not set(self.metadata).issubset(
            {
                'effective_backend',
                'fallback_category',
                'graph_version',
                'outcome',
                'permission_notice',
                'prompt_version',
                'status',
            }
        ):
            raise ValueError('V2 assistant metadata contains transient fields')
        object.__setattr__(
            self, 'metadata', MappingProxyType(dict(self.metadata))
        )
        if type(self.evidence) is not V1EvidenceProjection:
            raise ValueError('V2 assistant evidence projection is invalid')
        if type(self.hidden_match_count) is not int or self.hidden_match_count < 0:
            raise ValueError('V2 assistant hidden count is invalid')
        if not _lower_hmac(self.result_hmac):
            raise ValueError('V2 assistant result HMAC is invalid')
        assembled = _lower_hmac(self.assembled_answer_hmac)
        canned = self.canned_message_identity is not None
        if assembled == canned:
            raise ValueError('V2 assistant origin requires exactly one identity')
        if type(self.model_influence) is not tuple:
            raise ValueError('V2 assistant dependencies must be immutable')
        if assembled and not self.model_influence:
            raise ValueError('assembled V2 answer requires model influence')
        if canned and self.model_influence:
            raise ValueError('canned V2 answer cannot have model influence')


@dataclass(frozen=True, slots=True)
class LegacyAssistantMessageProjection:
    content: str
    metadata: Mapping[str, object]
    evidence: V1EvidenceProjection
    write_mode: Literal['legacy_trimmed'] = 'legacy_trimmed'


_ASSISTANT_EXACT_WRITE_SEAL = object()


@dataclass(frozen=True, slots=True)
class _AssistantExactWriteGrant:
    parent_agent_run_id: int
    conversation_id: int
    projection: AssistantMessageProjection = field(repr=False)
    parent_result_hmac: str
    _seal: object = field(repr=False, compare=False)


class AssistantExactWriteAuthority:
    """Unforgeable-by-constructor grant minted by the final product boundary."""

    __slots__ = ('_grant',)

    def __init__(self, grant: object) -> None:
        if (
            type(grant) is not _AssistantExactWriteGrant
            or grant._seal is not _ASSISTANT_EXACT_WRITE_SEAL
        ):
            raise TypeError('rag_v2_exact requires finalizer-minted authority')
        self._grant = grant

    def require(
        self,
        *,
        conversation: AssistantConversation,
        projection: AssistantMessageProjection,
    ) -> int:
        grant = self._grant
        if (
            projection is not grant.projection
            or conversation.id != grant.conversation_id
            or projection.result_hmac != grant.parent_result_hmac
        ):
            raise ValueError('rag_v2_exact authority does not match the product')
        return grant.parent_agent_run_id


def _mint_assistant_exact_write_authority(
    *,
    parent_agent_run_id: int,
    conversation_id: int,
    projection: AssistantMessageProjection,
    parent_result_hmac: str,
) -> AssistantExactWriteAuthority:
    if (
        type(parent_agent_run_id) is not int
        or parent_agent_run_id <= 0
        or type(conversation_id) is not int
        or conversation_id <= 0
        or type(projection) is not AssistantMessageProjection
        or not _lower_hmac(parent_result_hmac)
        or projection.result_hmac != parent_result_hmac
    ):
        raise ValueError('rag_v2_exact finalizer authority is invalid')
    return AssistantExactWriteAuthority(_AssistantExactWriteGrant(
        parent_agent_run_id=parent_agent_run_id,
        conversation_id=conversation_id,
        projection=projection,
        parent_result_hmac=parent_result_hmac,
        _seal=_ASSISTANT_EXACT_WRITE_SEAL,
    ))


class AssistantEvidenceWriter:
    """Flush assistant evidence rows without ever owning the transaction commit."""

    def __init__(
        self,
        *,
        fingerprint_secret: bytes,
        fingerprint_key_version: str,
        settings: Settings | None = None,
    ) -> None:
        if type(fingerprint_secret) is not bytes or not fingerprint_secret:
            raise ValueError('assistant fingerprint secret is unavailable')
        if type(fingerprint_key_version) is not str or not fingerprint_key_version:
            raise ValueError('assistant fingerprint key version is unavailable')
        self._secret = fingerprint_secret
        self._key_version = fingerprint_key_version
        self._verifier = fingerprint_key_material_verifier(
            fingerprint_secret.decode('utf-8')
        )
        self._settings = settings

    def append_final(
        self,
        *,
        db: Session,
        conversation: AssistantConversation,
        projection: AssistantMessageProjection,
        authority: AssistantExactWriteAuthority,
    ) -> AssistantMessage:
        if type(projection) is not AssistantMessageProjection:
            raise TypeError('server-issued V2 assistant projection is required')
        if type(authority) is not AssistantExactWriteAuthority:
            raise TypeError('rag_v2_exact requires finalizer-minted authority')
        parent_id = authority.require(
            conversation=conversation,
            projection=projection,
        )
        if type(conversation.id) is not int or conversation.id <= 0:
            raise ValueError('V2 assistant conversation is invalid')
        origin = (
            'rag_assembled'
            if projection.assembled_answer_hmac is not None
            else 'rag_canned'
        )
        origin_hmac = projection.assembled_answer_hmac or keyed_fingerprint(
            {'canned_message_identity': projection.canned_message_identity},
            secret=self._secret,
            schema_version='rag-canned-message-identity:v1',
            policy_version='rag-answer:v2',
        )
        dependencies = projection.model_influence
        message = AssistantMessage(
            conversation_id=conversation.id,
            role='assistant',
            content=projection.content,
            citations=[projection_record_to_transport(citation) for citation in projection.evidence.citations],
            source_ids=list(projection.evidence.source_ids),
            source_links=list(projection.evidence.source_links),
            source_snippets=list(projection.evidence.source_snippets),
            permission_level=projection.permission_level,
            hidden_match_count=projection.hidden_match_count,
            permission_notice=projection.permission_notice,
            agent_run_id=parent_id,
            evidence_contract_version=(
                'assistant-evidence:v1' if dependencies else 'none-v1'
            ),
            serving_dependency_count=len(dependencies),
            content_write_mode='rag_v2_exact',
            content_hmac_schema_version='assistant-message-content-hmac:v1',
            content_hmac_key_version=self._key_version,
            content_hmac_key_material_verifier=self._verifier,
            content_origin=origin,
            content_origin_hmac=origin_hmac,
            rag_result_hmac=projection.result_hmac,
            linked_agent_run_id=parent_id,
            dependency_set_hmac_schema_version=(
                'assistant-dependency-set-hmac:v2' if dependencies else None
            ),
            dependency_set_hmac=None,
            parent_selected_evidence_projection_hmac=(
                projection.evidence.projection_hmac if dependencies else None
            ),
            model_influence_set_hmac=None,
            metadata_=dict(projection.metadata),
        )
        conversation.updated_at = datetime.now(UTC)
        db.add(message)
        db.flush([message])
        message.assistant_message_content_hmac = keyed_fingerprint(
            {
                'agent_name_bytes': exact_utf8_bytes('rag_orchestrator_agent'),
                'assistant_message_id': message.id,
                'content_bytes': exact_utf8_bytes(message.content),
                'content_origin': origin,
                'content_origin_hmac': origin_hmac,
                'content_write_mode': 'rag_v2_exact',
                'conversation_id': conversation.id,
                'linked_agent_run_id': parent_id,
                'message_role': 'assistant',
                'prompt_version_bytes': exact_utf8_bytes('rag-answer:v2'),
                'rag_result_hmac': projection.result_hmac,
            },
            secret=self._secret,
            schema_version='assistant-message-content-hmac:v1',
            policy_version='assistant-evidence:v1',
        )
        if dependencies:
            if self._settings is None:
                raise ValueError('V2 dependency writer settings are unavailable')
            influence_hmac = build_model_influence_set_hmac(
                dependencies=dependencies, settings=self._settings
            )
            message.model_influence_set_hmac = influence_hmac
            dependency_rows = self._dependency_rows(
                db=db,
                message=message,
                projection=projection,
                dependencies=dependencies,
            )
            message.dependency_set_hmac = keyed_fingerprint(
                {
                    'assistant_message_content_hmac': (
                        message.assistant_message_content_hmac
                    ),
                    'content_origin': origin,
                    'content_origin_hmac': origin_hmac,
                    'content_write_mode': 'rag_v2_exact',
                    'dependencies': [
                        {
                            'candidate_ordinal': ordinal,
                            'dependency_child_hmac': row.dependency_child_hmac,
                        }
                        for ordinal, row in enumerate(dependency_rows)
                    ],
                    'dependency_count': len(dependencies),
                    'dependency_serving_scope': 'rag_v2',
                    'evidence_contract_version': 'assistant-evidence:v1',
                    'linked_agent_run_id': parent_id,
                    'model_influence_set_hmac': influence_hmac,
                    'parent_selected_evidence_projection_hmac': (
                        projection.evidence.projection_hmac
                    ),
                    'rag_result_hmac': projection.result_hmac,
                },
                secret=self._secret,
                schema_version='assistant-dependency-set-hmac:v2',
                policy_version='assistant-evidence:v1',
            )
            for row in dependency_rows:
                row.dependency_set_hmac = message.dependency_set_hmac
            db.add_all(dependency_rows)
            db.flush(dependency_rows)
            refs = []
            for row, dependency in zip(
                dependency_rows, dependencies, strict=True
            ):
                envelope = dependency.fresh_lookup_identity.version_envelope
                if type(envelope) is not TrustedServingVersionEnvelope:
                    continue
                provenance = envelope.provenance
                if type(provenance) is not ExplicitApprovalProvenance:
                    continue
                refs.extend(
                    AssistantMessageKnowledgeEvidenceRef(
                        dependency_id=row.id,
                        assistant_message_id=message.id,
                        approval_link_id=provenance.approval_link_id,
                        trusted_knowledge_evidence_link_id=(
                            link.trusted_knowledge_evidence_link_id
                        ),
                    )
                    for link in provenance.evidence_links
                )
            if refs:
                db.add_all(refs)
                db.flush(refs)
        db.flush([message])
        return message

    def _dependency_rows(
        self,
        *,
        db: Session,
        message: AssistantMessage,
        projection: AssistantMessageProjection,
        dependencies: tuple[ModelInfluenceDependencySnapshot, ...],
    ) -> list[AssistantMessageEvidenceDependency]:
        rows: list[AssistantMessageEvidenceDependency] = []
        selected_hmacs = iter(projection.evidence.citation_projection_hmacs)
        for ordinal, dependency in enumerate(dependencies):
            observation = dependency.observation
            identity = dependency.fresh_lookup_identity
            envelope = identity.version_envelope
            selected_hmac = None
            if dependency.dependency_role == 'selected_citation':
                try:
                    selected_hmac = next(selected_hmacs)
                except StopIteration:
                    raise ValueError(
                        'selected dependency projection is incomplete'
                    ) from None
                if not _lower_hmac(selected_hmac):
                    raise ValueError('selected citation HMAC is invalid')
            values, evidence_link_ids = _dependency_storage_values(envelope)
            child_hmac = keyed_fingerprint(
                {
                    'approval_link_id': values['approval_link_id'],
                    'approval_provenance_hmac': observation.approval_provenance_hmac,
                    'candidate_ordinal': ordinal,
                    'canonical_citation_projection_hmac': (
                        observation.canonical_citation_projection_hmac
                    ),
                    'dependency_kind': values['dependency_kind'],
                    'dependency_role': dependency.dependency_role,
                    'dependency_serving_scope': 'rag_v2',
                    'effective_permission': observation.effective_permission,
                    'evidence_link_ids': list(evidence_link_ids),
                    'evidence_link_set_hmac': observation.evidence_link_set_hmac,
                    'legacy_dependency_identity_hmac': None,
                    'model_content_hmac': observation.model_content_hmac,
                    'raw_document_chunk_id': values['document_chunk_id'],
                    'selected_v1_citation_projection_hmac': selected_hmac,
                    'serving_identity_hmac': observation.serving_identity_hmac,
                    'serving_version_fingerprint': (
                        observation.serving_version_fingerprint
                    ),
                    'support_mode': observation.support_mode,
                    'trusted_knowledge_id': values['knowledge_id'],
                },
                secret=self._secret,
                schema_version='assistant-dependency-child-hmac:v2',
                policy_version='assistant-evidence:v1',
            )
            row = AssistantMessageEvidenceDependency(
                    assistant_message_id=message.id,
                    candidate_ordinal=ordinal,
                    serving_document_id=identity.serving_document_id,
                    dependency_set_hmac='0' * 64,
                    serving_content_hash=observation.model_content_hmac,
                    permission_level=observation.effective_permission,
                    fingerprint_key_version=self._key_version,
                    fingerprint_key_material_verifier=self._verifier,
                    dependency_serving_scope='rag_v2',
                    dependency_role=dependency.dependency_role,
                    dependency_child_hmac=child_hmac,
                    approval_provenance_hmac=(
                        observation.approval_provenance_hmac
                    ),
                    evidence_link_set_hmac=observation.evidence_link_set_hmac,
                    legacy_dependency_identity_hmac=None,
                    model_content_hmac=observation.model_content_hmac,
                    canonical_citation_projection_hmac=(
                        observation.canonical_citation_projection_hmac
                    ),
                    selected_v1_citation_projection_hmac=selected_hmac,
                    serving_identity_hmac=observation.serving_identity_hmac,
                    serving_version_fingerprint=(
                        observation.serving_version_fingerprint
                    ),
                    support_mode=observation.support_mode,
                    **values,
            )
            if isinstance(db, Session):
                from backend.app.assistant.service import (
                    _knowledge_dependency_content_hash,
                    _raw_dependency_content_hash,
                )
                from backend.app.models import DocumentChunk, Source

                if row.dependency_kind == 'raw_chunk':
                    chunk = db.get(DocumentChunk, row.document_chunk_id)
                    source = db.get(Source, row.source_id)
                    content_hash = (
                        _raw_dependency_content_hash(
                            chunk=chunk,
                            source=source,
                            permission_level=row.permission_level,
                        )
                        if chunk is not None and source is not None
                        else None
                    )
                else:
                    content_hash = _knowledge_dependency_content_hash(db, row)
                if not _lower_hmac(content_hash):
                    raise ValueError(
                        'assistant dependency content authority is unavailable'
                    )
                row.serving_content_hash = content_hash
            rows.append(row)
        if next(selected_hmacs, None) is not None:
            raise ValueError('selected dependency projection has extra entries')
        return rows

    def append_legacy_evidence(
        self,
        *,
        db: Session,
        conversation: AssistantConversation,
        projection: LegacyAssistantMessageProjection,
    ) -> AssistantMessage:
        if type(projection) is not LegacyAssistantMessageProjection:
            raise TypeError('legacy assistant projection is required')
        content = projection.content.strip()
        if not content:
            raise ValueError('legacy assistant content must be nonblank')
        message = AssistantMessage(
            conversation_id=conversation.id,
            role='assistant',
            content=content,
            citations=list(projection.evidence.citations),
            source_ids=list(projection.evidence.source_ids),
            source_links=list(projection.evidence.source_links),
            source_snippets=list(projection.evidence.source_snippets),
            metadata_=dict(projection.metadata),
        )
        db.add(message)
        db.flush([message])
        return message


def _lower_hmac(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in '0123456789abcdef' for character in value)
    )


def _dependency_storage_values(
    envelope: object,
) -> tuple[dict[str, object], tuple[int, ...]]:
    common: dict[str, object] = {
        'document_chunk_id': None,
        'document_version_id': None,
        'source_id': None,
        'parser_run_id': None,
        'current_document_version_id': None,
        'server_content_signature_schema': None,
        'server_content_signature': None,
        'parser_policy_version': None,
        'parser_version': None,
        'chunk_policy_version': None,
        'knowledge_type': None,
        'knowledge_id': None,
        'approval_link_id': None,
        'legacy_human_base': False,
        'legacy_source_review_item_id': None,
    }
    if type(envelope) is RawServingVersionEnvelope:
        common.update(
            {
                'dependency_kind': 'raw_chunk',
                'document_chunk_id': envelope.document_chunk_id,
                'document_version_id': envelope.document_version_id,
                'source_id': envelope.source_row_id,
                'parser_run_id': envelope.parser_run_id,
                'current_document_version_id': (
                    envelope.current_document_version_id
                ),
                'server_content_signature_schema': (
                    envelope.server_content_signature_schema
                ),
                'server_content_signature': envelope.server_content_signature,
                'parser_policy_version': envelope.parser_policy_version,
                'parser_version': envelope.parser_version,
                'chunk_policy_version': envelope.chunk_policy_version,
            }
        )
        return common, ()
    if type(envelope) is not TrustedServingVersionEnvelope:
        raise ValueError('assistant dependency envelope is invalid')
    common.update(
        {
            'dependency_kind': 'trusted_knowledge',
            'knowledge_type': envelope.knowledge_type,
            'knowledge_id': envelope.knowledge_id,
        }
    )
    provenance = envelope.provenance
    if type(provenance) is ExplicitApprovalProvenance:
        common['approval_link_id'] = provenance.approval_link_id
        return common, tuple(
            sorted(
                link.trusted_knowledge_evidence_link_id
                for link in provenance.evidence_links
            )
        )
    if type(provenance) is LegacyHumanProvenance:
        common['legacy_human_base'] = True
        common['legacy_source_review_item_id'] = (
            provenance.legacy_source_review_item_id
        )
        return common, ()
    raise ValueError('assistant dependency provenance is invalid')


def _json_safe(value: object) -> object:
    if type(value) is tuple:
        return [_json_safe(item) for item in value]
    if type(value) is list:
        return [_json_safe(item) for item in value]
    if type(value) is dict:
        return {key: _json_safe(item) for key, item in value.items()}
    return value
