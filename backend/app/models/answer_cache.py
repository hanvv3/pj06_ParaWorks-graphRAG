from sqlalchemy import BigInteger, CheckConstraint, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base


class RagAnswerCacheEntry(Base):
    __tablename__ = 'rag_answer_cache_entries'
    __table_args__ = (
        CheckConstraint(
            'created_at >= 0 AND expires_at > created_at AND expires_at <= created_at + 86400',
            name='ck_answer_cache_retention',
        ),
        CheckConstraint(
            'length(key_hmac) = 64 AND length(scope_hmac) = 64 AND length(value_hmac) = 64',
            name='ck_answer_cache_hmac',
        ),
        Index('ix_answer_cache_expiry', 'expires_at'),
    )

    key_hmac: Mapped[str] = mapped_column(String(64), primary_key=True)
    scope_hmac: Mapped[str] = mapped_column(String(64), nullable=False)
    dependencies_json: Mapped[str] = mapped_column(Text, nullable=False)
    answer_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)
    expires_at: Mapped[int] = mapped_column(BigInteger, nullable=False)
    value_hmac: Mapped[str] = mapped_column(String(64), nullable=False)
