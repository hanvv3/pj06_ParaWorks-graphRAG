from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base


class AgentRun(Base):
    __tablename__ = 'agent_runs'
    __table_args__ = (
        UniqueConstraint(
            'id',
            'workflow_thread_id',
            name='uq_agent_runs_id_workflow',
        ),
        UniqueConstraint(
            'id',
            'workflow_thread_id',
            'agent_name',
            name='uq_agent_runs_id_workflow_agent',
        ),
        Index(
            'uq_agent_runs_workflow_effect',
            'workflow_thread_id',
            'effect_key',
            unique=True,
            postgresql_where=text(
                'workflow_thread_id IS NOT NULL AND effect_key IS NOT NULL'
            ),
            sqlite_where=text(
                'workflow_thread_id IS NOT NULL AND effect_key IS NOT NULL'
            ),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agent_name: Mapped[str] = mapped_column(String(64), index=True)
    prompt_version: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(32), default='complete', index=True)
    source_window: Mapped[str] = mapped_column(String(200), index=True)
    cache_key: Mapped[str] = mapped_column(String(128), index=True)
    model_name: Mapped[str] = mapped_column(String(128))
    generation_provider: Mapped[str | None] = mapped_column(String(120))
    generation_reasoning_effort: Mapped[str | None] = mapped_column(String(32))
    generation_route_version: Mapped[str | None] = mapped_column(String(64))
    generation_output_contract_version: Mapped[str | None] = mapped_column(
        String(64)
    )
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    permission_level: Mapped[str] = mapped_column(String(32), index=True)
    metadata_: Mapped[dict] = mapped_column('metadata', MutableDict.as_mutable(JSON), default=dict)
    workflow_thread_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    effect_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
