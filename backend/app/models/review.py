from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, Index, Integer, String, text
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base


class ReviewItem(Base):
    __tablename__ = 'review_items'
    __table_args__ = (
        Index(
            'uq_review_items_workflow_candidate',
            'workflow_thread_id',
            'candidate_key',
            unique=True,
            postgresql_where=text(
                'workflow_thread_id IS NOT NULL AND candidate_key IS NOT NULL'
            ),
            sqlite_where=text(
                'workflow_thread_id IS NOT NULL AND candidate_key IS NOT NULL'
            ),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_type: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict] = mapped_column(MutableDict.as_mutable(JSON))
    source_links: Mapped[list[str]] = mapped_column(MutableList.as_mutable(JSON), default=list)
    source_snippets: Mapped[list[str]] = mapped_column(MutableList.as_mutable(JSON), default=list)
    confidence_score: Mapped[float] = mapped_column(Float)
    permission_level: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(32), default='pending_review', index=True)
    reviewer_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    workflow_thread_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    candidate_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    predecessor_review_item_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
