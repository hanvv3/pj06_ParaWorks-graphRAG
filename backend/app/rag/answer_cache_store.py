"""Best-effort cache I/O in independent transactions; never masks authority reads."""

from __future__ import annotations

import hmac
import json
from time import time

from sqlalchemy import delete, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError

from backend.app.models.answer_cache import RagAnswerCacheEntry
from backend.app.rag.answer_cache import (
    ANSWER_CACHE_TTL_SECONDS,
    AnswerCacheHit,
    cache_hmac,
    eligible_answer_payload,
    exact_json,
    validate_cache_key,
)


class NullAnswerCache:
    def get(self, key, *, slots):
        return None

    def put(self, key, *, answer, slots):
        return False

    def cleanup(self, *, limit=100, strict=False):
        return 0


class PostgresAnswerCache:
    def __init__(self, *, engine, settings, validator, clock=time):
        if engine.dialect.name != 'postgresql':
            raise ValueError('answer cache requires PostgreSQL')
        self.engine, self.settings, self.validator, self.clock = (
            engine,
            settings,
            validator,
            clock,
        )

    def _now(self):
        value = self.clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError('invalid cache clock')
        return int(value)

    @staticmethod
    def _bound_transaction(conn):
        conn.execute(text("SET LOCAL lock_timeout = '500ms'"))
        conn.execute(text("SET LOCAL statement_timeout = '1s'"))

    def _signature(self, row):
        return cache_hmac(
            {
                name: row[name]
                for name in (
                    'key_hmac',
                    'scope_hmac',
                    'dependencies_json',
                    'answer_json',
                    'created_at',
                    'expires_at',
                )
            },
            settings=self.settings,
            domain='rag-answer-cache-value:v1',
        )

    def get(self, key, *, slots):
        try:
            validate_cache_key(key, slots=slots, settings=self.settings)
            now = self._now()
        except (ValueError, TypeError, AttributeError):
            return None
        table = RagAnswerCacheEntry.__table__
        try:
            with self.engine.begin() as conn:
                self._bound_transaction(conn)
                row = (
                    conn.execute(
                        select(table).where(
                            table.c.key_hmac == key.key_hmac,
                            table.c.scope_hmac == key.scope_hmac,
                            table.c.expires_at > now,
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
        except SQLAlchemyError:
            return None
        if row is None:
            return None
        try:
            if (
                row['dependencies_json'] != key.dependencies_json
                or not 0 <= row['created_at'] <= now < row['expires_at']
                or not 0
                < row['expires_at'] - row['created_at']
                <= ANSWER_CACHE_TTL_SECONDS
                or not hmac.compare_digest(row['value_hmac'], self._signature(row))
            ):
                return None
            answer = self.validator.validate(
                json.loads(row['answer_json']), slots=slots
            )
            if (
                not answer.blocks
                or answer.insufficient_reason is not None
                or not row['created_at'] <= self._now() < row['expires_at']
            ):
                return None
            return AnswerCacheHit(
                answer,
                key.key_hmac,
                key.scope_hmac,
                row['value_hmac'],
                row['created_at'],
                row['expires_at'],
            )
        except (ValueError, TypeError, AttributeError):
            return None

    def put(self, key, *, answer, slots):
        try:
            validate_cache_key(key, slots=slots, settings=self.settings)
            payload = eligible_answer_payload(
                answer, slots=slots, validator=self.validator
            )
            if payload is None:
                return False
            now = self._now()
            row = {
                'key_hmac': key.key_hmac,
                'scope_hmac': key.scope_hmac,
                'dependencies_json': key.dependencies_json,
                'answer_json': exact_json(payload),
                'created_at': now,
                'expires_at': now + ANSWER_CACHE_TTL_SECONDS,
            }
            row['value_hmac'] = self._signature(row)
        except (ValueError, TypeError, AttributeError):
            return False
        table = RagAnswerCacheEntry.__table__
        try:
            with self.engine.begin() as conn:
                self._bound_transaction(conn)
                # Only expired entries may be replaced by a freshly validated answer.
                result = conn.execute(
                    insert(table)
                    .values(**row)
                    .on_conflict_do_update(
                        index_elements=[table.c.key_hmac],
                        set_=row,
                        where=table.c.expires_at <= now,
                    )
                    .returning(table.c.key_hmac)
                )
                return result.scalar_one_or_none() == key.key_hmac
        except SQLAlchemyError:
            return False

    def cleanup(self, *, limit=100, strict=False):
        """Strict operator calls must distinguish database failure from drained."""
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError('cleanup limit must be between 1 and 1000')
        table = RagAnswerCacheEntry.__table__
        try:
            now = self._now()
            with self.engine.begin() as conn:
                self._bound_transaction(conn)
                expired = (
                    select(table.c.key_hmac)
                    .where(table.c.expires_at <= now)
                    .order_by(table.c.expires_at, table.c.key_hmac)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
                return conn.execute(
                    delete(table).where(table.c.key_hmac.in_(expired))
                ).rowcount
        except SQLAlchemyError:
            if strict:
                raise
            return 0


def create_answer_cache(*, engine, settings, validator, enabled=False, clock=time):
    if enabled is not True or engine.dialect.name != 'postgresql':
        return NullAnswerCache()
    return PostgresAnswerCache(
        engine=engine, settings=settings, validator=validator, clock=clock
    )
