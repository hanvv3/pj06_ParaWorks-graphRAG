from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.assistant_legacy_guards import register_sqlite_guards
from backend.app.db.base import Base

register_sqlite_guards(Base.metadata)


class AssistantConversation(Base):
    __tablename__ = 'assistant_conversations'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(120), index=True)
    title: Mapped[str] = mapped_column(String(160), default='새 대화')
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        index=True,
    )
    messages: Mapped[list['AssistantMessage']] = relationship(
        back_populates='conversation',
        cascade='all, delete-orphan',
        order_by='AssistantMessage.created_at, AssistantMessage.id',
    )


class AssistantMessage(Base):
    __tablename__ = 'assistant_messages'
    __table_args__ = (
        CheckConstraint(
            "evidence_contract_version IS NULL OR evidence_contract_version IN "
            "('none-v1', 'assistant-evidence:v1')",
            name='ck_assistant_messages_evidence_contract',
        ),
        CheckConstraint(
            'serving_dependency_count IS NULL OR serving_dependency_count >= 0',
            name='ck_assistant_messages_serving_dependency_count',
        ),
        CheckConstraint(
            "dependency_set_hmac_schema_version IS NULL OR "
            "dependency_set_hmac_schema_version IN ('assistant-dependency-set-hmac:v2', "
            "'assistant-dependency-set-hmac:v3')",
            name='ck_assistant_messages_dependency_set_schema',
        ),
        CheckConstraint(
            "content_write_mode IS NULL OR content_write_mode IN "
            "('legacy_trimmed', 'rag_v2_exact')",
            name='ck_assistant_messages_content_write_mode',
        ),
        CheckConstraint(
            "content_origin IS NULL OR content_origin IN "
            "('rag_assembled', 'rag_canned', 'legacy_evidence')",
            name='ck_assistant_messages_content_origin',
        ),
        CheckConstraint(
            'CASE WHEN content_write_mode IS NULL THEN '
            '(content_hmac_schema_version IS NULL AND '
            'assistant_message_content_hmac IS NULL AND '
            'content_hmac_key_version IS NULL AND '
            'content_hmac_key_material_verifier IS NULL AND content_origin IS NULL AND '
            'content_origin_hmac IS NULL AND rag_result_hmac IS NULL AND '
            'linked_agent_run_id IS NULL AND dependency_set_hmac_schema_version IS NULL AND '
            'dependency_set_hmac IS NULL AND '
            'parent_selected_evidence_projection_hmac IS NULL AND '
            'model_influence_set_hmac IS NULL) ELSE '
            "(content_hmac_schema_version IS NOT NULL AND "
            "content_hmac_schema_version = 'assistant-message-content-hmac:v1' AND "
            'assistant_message_content_hmac IS NOT NULL AND '
            'length(assistant_message_content_hmac) = 64 AND '
            'content_hmac_key_version IS NOT NULL AND '
            'content_hmac_key_material_verifier IS NOT NULL AND '
            'length(content_hmac_key_material_verifier) = 64 AND '
            'content_origin IS NOT NULL AND content_origin_hmac IS NOT NULL AND '
            'length(content_origin_hmac) = 64 AND '
            "((content_write_mode = 'rag_v2_exact' AND "
            "content_origin IN ('rag_assembled', 'rag_canned') AND "
            'rag_result_hmac IS NOT NULL AND length(rag_result_hmac) = 64 AND '
            'linked_agent_run_id IS NOT NULL) OR '
            "(content_write_mode = 'legacy_trimmed' AND "
            "content_origin = 'legacy_evidence' AND rag_result_hmac IS NULL AND "
            'linked_agent_run_id IS NULL))) END',
            name='ck_assistant_messages_content_integrity',
        ),
        CheckConstraint(
            "content_write_mode IS NULL OR ((content_write_mode = 'rag_v2_exact' AND "
            "content_origin = 'rag_canned' AND evidence_contract_version IS NOT NULL AND "
            "evidence_contract_version = 'none-v1' AND serving_dependency_count IS NOT NULL "
            'AND serving_dependency_count = 0 AND '
            'dependency_set_hmac_schema_version IS NULL AND dependency_set_hmac IS NULL AND '
            'parent_selected_evidence_projection_hmac IS NULL AND '
            'model_influence_set_hmac IS NULL) OR '
            "(content_write_mode = 'rag_v2_exact' AND content_origin = 'rag_assembled' AND "
            "evidence_contract_version IS NOT NULL AND evidence_contract_version = "
            "'assistant-evidence:v1' AND serving_dependency_count IS NOT NULL AND "
            'serving_dependency_count > 0 AND '
            'dependency_set_hmac_schema_version IS NOT NULL AND '
            "dependency_set_hmac_schema_version = 'assistant-dependency-set-hmac:v2' AND "
            'dependency_set_hmac IS NOT NULL AND length(dependency_set_hmac) = 64 AND '
            'parent_selected_evidence_projection_hmac IS NOT NULL AND '
            'length(parent_selected_evidence_projection_hmac) = 64 AND '
            'model_influence_set_hmac IS NOT NULL AND '
            'length(model_influence_set_hmac) = 64) OR '
            "(content_write_mode = 'legacy_trimmed' AND content_origin = 'legacy_evidence' "
            'AND evidence_contract_version IS NOT NULL AND evidence_contract_version = '
            "'assistant-evidence:v1' AND serving_dependency_count IS NOT NULL AND "
            'serving_dependency_count > 0 AND '
            'dependency_set_hmac_schema_version IS NOT NULL AND '
            "dependency_set_hmac_schema_version IN ('assistant-dependency-set-hmac:v2', "
            "'assistant-dependency-set-hmac:v3') AND "
            'dependency_set_hmac IS NOT NULL AND length(dependency_set_hmac) = 64 AND '
            'parent_selected_evidence_projection_hmac IS NOT NULL AND '
            'length(parent_selected_evidence_projection_hmac) = 64 AND '
            'model_influence_set_hmac IS NULL))',
            name='ck_assistant_messages_content_origin_xor',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey('assistant_conversations.id'), index=True)
    role: Mapped[str] = mapped_column(String(24), index=True)
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[list] = mapped_column(MutableList.as_mutable(JSON), default=list)
    source_ids: Mapped[list] = mapped_column(MutableList.as_mutable(JSON), default=list)
    source_links: Mapped[list] = mapped_column(MutableList.as_mutable(JSON), default=list)
    source_snippets: Mapped[list] = mapped_column(MutableList.as_mutable(JSON), default=list)
    permission_level: Mapped[str | None] = mapped_column(String(32), nullable=True)
    hidden_match_count: Mapped[int] = mapped_column(Integer, default=0)
    permission_notice: Mapped[str | None] = mapped_column(Text, nullable=True)
    agent_run_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    evidence_contract_version: Mapped[str | None] = mapped_column(String(32))
    serving_dependency_count: Mapped[int | None] = mapped_column(Integer)
    content_write_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)
    content_hmac_schema_version: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    assistant_message_content_hmac: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    content_hmac_key_version: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    content_hmac_key_material_verifier: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    content_origin: Mapped[str | None] = mapped_column(String(32), nullable=True)
    content_origin_hmac: Mapped[str | None] = mapped_column(String(64), nullable=True)
    rag_result_hmac: Mapped[str | None] = mapped_column(String(64), nullable=True)
    linked_agent_run_id: Mapped[int | None] = mapped_column(
        ForeignKey('agent_runs.id', ondelete='RESTRICT'), nullable=True, index=True
    )
    dependency_set_hmac_schema_version: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    dependency_set_hmac: Mapped[str | None] = mapped_column(String(64), nullable=True)
    parent_selected_evidence_projection_hmac: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    model_influence_set_hmac: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    metadata_: Mapped[dict] = mapped_column('metadata', MutableDict.as_mutable(JSON), default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        index=True,
    )
    conversation: Mapped[AssistantConversation] = relationship(back_populates='messages')
