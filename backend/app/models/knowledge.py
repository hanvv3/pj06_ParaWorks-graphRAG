from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, Index, Integer, String, Text, text
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base


class Project(Base):
    __tablename__ = 'projects'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(300))
    summary: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

class DecisionRecord(Base):
    __tablename__ = 'decision_records'
    __table_args__ = (
        Index(
            'uq_decision_records_source_review_item',
            'source_review_item_id',
            unique=True,
            postgresql_where=text('source_review_item_id IS NOT NULL'),
            sqlite_where=text('source_review_item_id IS NOT NULL'),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_key: Mapped[str | None] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(300))
    decision_summary: Mapped[str] = mapped_column(Text)
    source_links: Mapped[list[str]] = mapped_column(MutableList.as_mutable(JSON), default=list)
    source_snippets: Mapped[list[str]] = mapped_column(MutableList.as_mutable(JSON), default=list)
    confidence_score: Mapped[float] = mapped_column(Float)
    permission_level: Mapped[str] = mapped_column(String(32), index=True)
    review_status: Mapped[str] = mapped_column(String(32), default='pending_review')
    source_review_item_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class HistoryEvent(Base):
    __tablename__ = 'history_events'
    __table_args__ = (
        Index(
            'uq_history_events_source_review_item',
            'source_review_item_id',
            unique=True,
            postgresql_where=text('source_review_item_id IS NOT NULL'),
            sqlite_where=text('source_review_item_id IS NOT NULL'),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_key: Mapped[str | None] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(300))
    reason: Mapped[str] = mapped_column(Text)
    source_links: Mapped[list[str]] = mapped_column(MutableList.as_mutable(JSON), default=list)
    source_snippets: Mapped[list[str]] = mapped_column(MutableList.as_mutable(JSON), default=list)
    confidence_score: Mapped[float] = mapped_column(Float)
    permission_level: Mapped[str] = mapped_column(String(32), index=True)
    review_status: Mapped[str] = mapped_column(String(32), default='pending_review')
    source_review_item_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class TimelineEvent(Base):
    __tablename__ = 'timeline_events'
    __table_args__ = (
        Index(
            'uq_timeline_events_source_review_item',
            'source_review_item_id',
            unique=True,
            postgresql_where=text('source_review_item_id IS NOT NULL'),
            sqlite_where=text('source_review_item_id IS NOT NULL'),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_key: Mapped[str | None] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(300))
    result_summary: Mapped[str] = mapped_column(Text)
    source_links: Mapped[list[str]] = mapped_column(MutableList.as_mutable(JSON), default=list)
    source_snippets: Mapped[list[str]] = mapped_column(MutableList.as_mutable(JSON), default=list)
    confidence_score: Mapped[float] = mapped_column(Float)
    permission_level: Mapped[str] = mapped_column(String(32), index=True)
    review_status: Mapped[str] = mapped_column(String(32), default='pending_review')
    source_review_item_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class Todo(Base):
    __tablename__ = 'todos'
    __table_args__ = (
        Index(
            'uq_todos_source_review_item',
            'source_review_item_id',
            unique=True,
            postgresql_where=text('source_review_item_id IS NOT NULL'),
            sqlite_where=text('source_review_item_id IS NOT NULL'),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_key: Mapped[str | None] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(300))
    assignee: Mapped[str | None] = mapped_column(String(300))
    due_date: Mapped[str | None] = mapped_column(String(32), index=True)
    priority: Mapped[str] = mapped_column(String(32))
    priority_reason: Mapped[str] = mapped_column(Text)
    source_links: Mapped[list[str]] = mapped_column(MutableList.as_mutable(JSON), default=list)
    source_snippets: Mapped[list[str]] = mapped_column(MutableList.as_mutable(JSON), default=list)
    confidence_score: Mapped[float] = mapped_column(Float)
    permission_level: Mapped[str] = mapped_column(String(32), index=True)
    review_status: Mapped[str] = mapped_column(String(32), default='pending_review')
    source_review_item_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    completed_by: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
