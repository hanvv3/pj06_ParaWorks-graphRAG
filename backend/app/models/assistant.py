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

from backend.app.db.base import Base


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
    metadata_: Mapped[dict] = mapped_column('metadata', MutableDict.as_mutable(JSON), default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        index=True,
    )
    conversation: Mapped[AssistantConversation] = relationship(back_populates='messages')
