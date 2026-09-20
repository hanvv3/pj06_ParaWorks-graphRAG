from __future__ import annotations

import importlib.util
import inspect as pyinspect
import math
import os
import struct
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import IntegrityError

from backend.app.core.config import get_settings

REVISION = 'd1a2b3c4e5f6'
PREVIOUS_REVISION = '9d7f3a1c6e20'
MIGRATION_PATH = Path(
    'backend/migrations/versions/d1a2b3c4e5f6_add_rag_serving_projection.py'
)

LEGACY_VECTOR_COLUMNS = {
    'id',
    'document_id',
    'embedding_model',
    'embedding_dimensions',
    'content_hash',
    'status',
    'last_error',
    'indexed_at',
}
D_VECTOR_COLUMNS = {
    'corpus_generation_id',
    'serving_kind',
    'support_mode',
    'effective_permission',
    'serving_identity_hmac',
    'serving_version_fingerprint',
    'model_content_hmac',
    'canonical_citation_projection_hmac',
    'index_policy_version',
    'pgvector_cosine_policy_version',
    'cosine_indexable',
    'vector_index_generation',
    'vector_index_state_hmac',
    'fingerprint_key_version',
    'fingerprint_key_material_verifier',
}
CORPUS_COLUMNS = {
    'id',
    'corpus_generation',
    'vector_index_generation',
    'embedding_model',
    'embedding_dimensions',
    'index_policy_version',
    'pgvector_cosine_policy_version',
    'fingerprint_key_version',
    'fingerprint_key_material_verifier',
    'updated_at',
}
LEXICAL_COLUMNS = {
    'id',
    'corpus_generation_id',
    'corpus_generation',
    'serving_document_id',
    'serving_kind',
    'support_mode',
    'effective_permission',
    'serving_identity_hmac',
    'serving_version_fingerprint',
    'model_content_hmac',
    'canonical_citation_projection_hmac',
    'title_lower',
    'searchable_lower',
    'lexical_contract_version',
    'fingerprint_key_version',
    'fingerprint_key_material_verifier',
    'created_at',
    'updated_at',
}
CORPUS_CHECKS = {
    'ck_rag_serving_corpus_singleton',
    'ck_rag_serving_corpus_generations_nonnegative',
    'ck_rag_serving_corpus_policy_identity',
    'ck_rag_serving_corpus_key_identity',
}
LEXICAL_CHECKS = {
    'ck_rag_lexical_serving_identity',
    'ck_rag_lexical_serving_hmacs',
    'ck_rag_lexical_serving_contract',
    'ck_rag_lexical_serving_permission_support',
    'ck_rag_lexical_serving_generation',
}
VECTOR_CHECKS = {
    'ck_vector_index_state_d_provenance',
    'ck_vector_index_state_d_policy',
    'ck_vector_index_state_d_hmacs',
}
LEXICAL_INDEXES = {
    'ix_rag_lexical_serving_projection_permission',
    'ix_rag_lexical_serving_projection_searchable_lower',
    'ix_rag_lexical_serving_projection_support_mode',
}
LEXICAL_UNIQUES = {
    'uq_rag_lexical_serving_projection_document',
    'uq_rag_lexical_serving_projection_identity_hmac',
}


@pytest.fixture
def sqlite_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Config, str]]:
    database_url = f'sqlite:///{(tmp_path / "rag-v2-migration.db").as_posix()}'
    monkeypatch.setenv('PARAWORKS_DEMO_MODE', 'false')
    monkeypatch.setenv('PARAWORKS_DATABASE_URL', database_url)
    get_settings.cache_clear()
    try:
        yield Config('alembic.ini'), database_url
    finally:
        get_settings.cache_clear()


def _load_migration_module():
    assert MIGRATION_PATH.is_file()
    spec = importlib.util.spec_from_file_location(
        'rag_v2_task3_migration', MIGRATION_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _create_pinned_pre_d_schema(config: Config, database_url: str) -> Engine:
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                'CREATE TABLE vector_index_states ('
                'id INTEGER PRIMARY KEY, document_id VARCHAR(200) NOT NULL, '
                'embedding_model VARCHAR(120) NOT NULL, embedding_dimensions INTEGER NOT NULL, '
                'content_hash VARCHAR(64) NOT NULL, status VARCHAR(32) NOT NULL, '
                'last_error TEXT, indexed_at DATETIME NOT NULL)'
            )
        )
    command.stamp(config, PREVIOUS_REVISION)
    return engine


def _names(items: list[dict]) -> set[str | None]:
    return {item['name'] for item in items}


def _foreign_key_contract(schema, table_name: str) -> set[tuple]:
    return {
        (
            item['name'],
            tuple(item['constrained_columns']),
            item['referred_table'],
            tuple(item['referred_columns']),
            item['options'].get('ondelete'),
        )
        for item in schema.get_foreign_keys(table_name)
    }


def _insert_corpus(connection) -> None:
    connection.execute(
        text(
            'INSERT INTO rag_serving_corpus_generations '
            '(id, corpus_generation, vector_index_generation, embedding_model, '
            'embedding_dimensions, index_policy_version, pgvector_cosine_policy_version, '
            'fingerprint_key_version, fingerprint_key_material_verifier, updated_at) '
            "VALUES (1, 0, 0, 'text-embedding-3-small', 1536, "
            "'rag-v2-serving-index:v1', 'pgvector-cosine-indexable:v1', "
            "'runtime-key-v1', :verifier, CURRENT_TIMESTAMP)"
        ),
        {'verifier': 'a' * 64},
    )


def _projection_parameters(**overrides):
    values = {
        'document_id': 'chunk:1',
        'identity_hmac': 'b' * 64,
        'version_hmac': 'c' * 64,
        'content_hmac': 'd' * 64,
        'citation_hmac': 'e' * 64,
        'permission': 'internal',
        'support': 'source_observation',
    }
    values.update(overrides)
    return values


def _insert_projection(connection, **overrides) -> None:
    connection.execute(
        text(
            'INSERT INTO rag_lexical_serving_projections '
            '(corpus_generation_id, corpus_generation, serving_document_id, '
            'serving_kind, support_mode, effective_permission, serving_identity_hmac, '
            'serving_version_fingerprint, model_content_hmac, '
            'canonical_citation_projection_hmac, title_lower, searchable_lower, '
            'lexical_contract_version, fingerprint_key_version, '
            'fingerprint_key_material_verifier, created_at, updated_at) VALUES '
            "(1, 0, :document_id, 'raw_chunk', :support, :permission, :identity_hmac, "
            ":version_hmac, :content_hmac, :citation_hmac, 'title', "
            "'title\\nbody', 'rag-keyword-lexical-compat:v1', 'runtime-key-v1', "
            ':verifier, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)'
        ),
        {**_projection_parameters(**overrides), 'verifier': 'a' * 64},
    )


def test_first_d_revision_has_exact_chain() -> None:
    migration = _load_migration_module()
    assert migration.revision == REVISION
    assert migration.down_revision == PREVIOUS_REVISION


def test_sqlite_exact_revision_upgrade_defines_the_complete_d_schema(
    sqlite_migration,
) -> None:
    config, database_url = sqlite_migration
    engine = _create_pinned_pre_d_schema(config, database_url)

    command.upgrade(config, REVISION)

    schema = inspect(engine)
    assert set(schema.get_table_names()) == {
        'alembic_version',
        'vector_index_states',
        'rag_serving_corpus_generations',
        'rag_lexical_serving_projections',
    }
    assert _names(schema.get_columns('rag_serving_corpus_generations')) == CORPUS_COLUMNS
    assert _names(schema.get_columns('rag_lexical_serving_projections')) == (
        LEXICAL_COLUMNS
    )
    assert _names(schema.get_columns('vector_index_states')) == (
        LEGACY_VECTOR_COLUMNS | D_VECTOR_COLUMNS
    )
    assert _names(
        schema.get_check_constraints('rag_serving_corpus_generations')
    ) == CORPUS_CHECKS
    assert _names(
        schema.get_check_constraints('rag_lexical_serving_projections')
    ) == LEXICAL_CHECKS
    assert _names(schema.get_check_constraints('vector_index_states')) == VECTOR_CHECKS
    assert _names(schema.get_indexes('rag_lexical_serving_projections')) == (
        LEXICAL_INDEXES
    )
    assert _names(schema.get_unique_constraints('rag_lexical_serving_projections')) == (
        LEXICAL_UNIQUES
    )
    assert _foreign_key_contract(schema, 'rag_lexical_serving_projections') == {
        (
            'fk_rag_lexical_serving_projection_corpus',
            ('corpus_generation_id',),
            'rag_serving_corpus_generations',
            ('id',),
            'RESTRICT',
        )
    }
    assert _foreign_key_contract(schema, 'vector_index_states') == {
        (
            'fk_vector_index_state_rag_serving_corpus',
            ('corpus_generation_id',),
            'rag_serving_corpus_generations',
            ('id',),
            'RESTRICT',
        )
    }
    with engine.connect() as connection:
        assert (
            connection.scalar(text('SELECT version_num FROM alembic_version'))
            == REVISION
        )


def test_sqlite_exact_revision_downgrade_removes_only_additive_d_schema(
    sqlite_migration,
) -> None:
    config, database_url = sqlite_migration
    engine = _create_pinned_pre_d_schema(config, database_url)
    command.upgrade(config, REVISION)

    command.downgrade(config, PREVIOUS_REVISION)

    schema = inspect(engine)
    assert set(schema.get_table_names()) == {'alembic_version', 'vector_index_states'}
    assert _names(schema.get_columns('vector_index_states')) == LEGACY_VECTOR_COLUMNS
    assert schema.get_check_constraints('vector_index_states') == []
    assert schema.get_foreign_keys('vector_index_states') == []
    with engine.connect() as connection:
        assert (
            connection.scalar(text('SELECT version_num FROM alembic_version'))
            == PREVIOUS_REVISION
        )


def test_sqlite_upgrade_preserves_historical_vector_without_d_retagging(
    sqlite_migration,
) -> None:
    config, database_url = sqlite_migration
    engine = _create_pinned_pre_d_schema(config, database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                'INSERT INTO vector_index_states VALUES '
                "(1, 'legacy:1', 'legacy:8', 8, :content_hash, 'indexed', NULL, CURRENT_TIMESTAMP)"
            ),
            {'content_hash': '0' * 64},
        )

    command.upgrade(config, REVISION)

    with engine.connect() as connection:
        row = connection.execute(
            text(
                'SELECT serving_kind, serving_identity_hmac, index_policy_version, '
                'vector_index_generation, vector_index_state_hmac '
                'FROM vector_index_states WHERE id=1'
            )
        ).one()
    assert tuple(row) == (None, None, None, None, None)


def test_sqlite_exact_revision_refuses_downgrade_with_retained_serving_state(
    sqlite_migration,
) -> None:
    config, database_url = sqlite_migration
    engine = _create_pinned_pre_d_schema(config, database_url)
    command.upgrade(config, REVISION)
    with engine.begin() as connection:
        _insert_corpus(connection)

    with pytest.raises(RuntimeError, match='corpus generation state is retained'):
        command.downgrade(config, PREVIOUS_REVISION)

    schema = inspect(engine)
    assert 'rag_serving_corpus_generations' in schema.get_table_names()
    with engine.connect() as connection:
        assert connection.scalar(
            text('SELECT version_num FROM alembic_version')
        ) == REVISION
        assert connection.scalar(
            text('SELECT count(*) FROM rag_serving_corpus_generations')
        ) == 1


def test_exact_revision_upgrade_is_additive_and_has_no_backfill_or_trigger() -> None:
    migration = _load_migration_module()
    reachable_upgrade_source = '\n'.join(
        pyinspect.getsource(function)
        for function in (
            migration.upgrade,
            migration._create_model_table,
            migration._add_vector_columns,
            migration._add_vector_constraints,
            migration._create_check,
            migration._create_fk,
            migration._install_postgresql_scorers,
        )
    ).upper()
    for prohibited in (
        'DROP TABLE',
        'DROP COLUMN',
        'DROP CONSTRAINT',
        'CREATE TRIGGER',
        'UPDATE ',
        'DELETE ',
        'INSERT ',
    ):
        assert prohibited not in reachable_upgrade_source


def test_exact_revision_installs_only_the_named_scorer_sql_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = _load_migration_module()
    statements: list[str] = []
    bind = SimpleNamespace(dialect=SimpleNamespace(name='postgresql'))
    monkeypatch.setattr(
        migration,
        'op',
        SimpleNamespace(
            get_bind=lambda: bind,
            execute=lambda statement: statements.append(str(statement)),
        ),
    )

    migration._install_postgresql_scorers()

    assert len(statements) == 2
    rounder_sql, scorer_sql = (' '.join(statement.split()) for statement in statements)
    assert rounder_sql.startswith(
        'CREATE OR REPLACE FUNCTION rag_python_round6_binary64_v1( '
        'value double precision ) RETURNS double precision'
    )
    assert scorer_sql.startswith(
        'CREATE OR REPLACE FUNCTION rag_python_lexical_score_v1( '
        'title_lower text, searchable_lower text, query_terms text[], phrase_lower text '
        ') RETURNS TABLE(score double precision, matched_terms text[])'
    )
    assert 'IF biased_exponent = 2047 THEN RETURN value; END IF;' in rounder_sql
    assert "RETURN '-0'::double precision" in rounder_sql
    assert 'remainder * 2 = denominator AND mod(rounded_integer, 2) = 1' in (
        rounder_sql
    )
    assert 'unnest(query_terms) WITH ORDINALITY' in scorer_sql
    assert 'array_agg(term ORDER BY ordinal)' in scorer_sql
    assert 'cardinality(query_terms) = 0 OR matched_count = 0 THEN 0.0' in (
        scorer_sql
    )
    assert 'strpos(searchable_lower, phrase_lower) > 0' in scorer_sql
    assert 'LEAST(title_hits::double precision * 0.15, 0.45)' in scorer_sql
    assert 'rag_python_round6_binary64_v1(' in scorer_sql
    combined_sql = ' '.join(statements).upper()
    for prohibited in ('CREATE TRIGGER', 'UPDATE ', 'DELETE ', 'INSERT '):
        assert prohibited not in combined_sql


@pytest.fixture
def postgres_migration(monkeypatch: pytest.MonkeyPatch) -> Iterator[Engine]:
    original_url = os.getenv('PARAWORKS_TEST_POSTGRES_URL')
    if not original_url:
        pytest.skip(
            'PARAWORKS_TEST_POSTGRES_URL is unavailable for Task 3 PostgreSQL checks'
        )
    parsed = make_url(original_url)
    if parsed.host != '127.0.0.1' or parsed.port != 55432:
        pytest.fail('Task 3 PostgreSQL checks require disposable 127.0.0.1:55432')
    admin = create_engine(original_url)
    schema_name = f'rag_task3_{uuid4().hex}'
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA {schema_name}'))
        connection.execute(
            text(
                f'CREATE TABLE {schema_name}.alembic_version '
                '(version_num VARCHAR(32) NOT NULL PRIMARY KEY)'
            )
        )
    query = dict(parsed.query)
    query['options'] = f'-csearch_path={schema_name},public'
    isolated_url = parsed.set(query=query).render_as_string(hide_password=False)
    monkeypatch.setenv('PARAWORKS_DEMO_MODE', 'false')
    monkeypatch.setenv('DATABASE_URL', isolated_url)
    monkeypatch.setenv('PARAWORKS_DATABASE_URL', isolated_url)
    get_settings.cache_clear()
    command.upgrade(Config('alembic.ini'), 'head')
    engine = create_engine(isolated_url)
    try:
        yield engine
    finally:
        engine.dispose()
        get_settings.cache_clear()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA {schema_name} CASCADE'))
        admin.dispose()


def test_postgresql_enforces_serving_and_vector_projection_contracts(
    postgres_migration,
) -> None:
    engine = postgres_migration
    with engine.begin() as connection:
        _insert_corpus(connection)
        _insert_projection(connection)

    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            text(
                'INSERT INTO rag_serving_corpus_generations '
                '(id, corpus_generation, vector_index_generation, embedding_model, '
                'embedding_dimensions, index_policy_version, '
                'pgvector_cosine_policy_version, fingerprint_key_version, '
                'fingerprint_key_material_verifier, updated_at) VALUES '
                "(2, 0, 0, 'model', 8, 'rag-v2-serving-index:v1', "
                "'pgvector-cosine-indexable:v1', 'key', :verifier, CURRENT_TIMESTAMP)"
            ),
            {'verifier': 'a' * 64},
        )
    with pytest.raises(IntegrityError), engine.begin() as connection:
        _insert_projection(connection, identity_hmac='2' * 64)
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            text(
                'UPDATE rag_serving_corpus_generations SET corpus_generation=-1 WHERE id=1'
            )
        )
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            text(
                'INSERT INTO vector_index_states '
                '(document_id, embedding_model, embedding_dimensions, content_hash, status, '
                "indexed_at, serving_kind) VALUES ('partial:1', 'model', 8, :content_hash, "
                "'indexed', CURRENT_TIMESTAMP, 'raw_chunk')"
            ),
            {'content_hash': '0' * 64},
        )


def test_postgresql_fresh_schema_reaches_head(postgres_migration) -> None:
    with postgres_migration.connect() as connection:
        assert connection.scalar(text('SELECT version_num FROM alembic_version')) == (
            'd7a8b9c0d1e2'
        )


def test_postgresql_installs_exact_python_lexical_scorer_functions(
    postgres_migration,
) -> None:
    engine = postgres_migration
    with engine.connect() as connection:
        rounder = connection.scalar(
            text(
                "SELECT to_regprocedure('rag_python_round6_binary64_v1(double precision)')"
            )
        )
        scorer = connection.scalar(
            text(
                "SELECT to_regprocedure('rag_python_lexical_score_v1(text,text,text[],text)')"
            )
        )
        lexical_cases = (
            ('title', 'title body', [], 'title', 0.0, []),
            ('title', 'title body', ['absent'], 'absent', 0.0, []),
            (
                'redis queue migration',
                'redis queue migration\nredis remains searchable',
                ['redis', 'redis', 'queue', 'absent'],
                'redis queue',
                round(3 / 4 + 1.0 + 0.45, 6),
                ['redis', 'redis', 'queue'],
            ),
            (
                'unrelated',
                'redis before queue',
                ['queue', 'redis'],
                'not present',
                1.0,
                ['queue', 'redis'],
            ),
            (
                'alpha beta gamma delta',
                'alpha beta gamma delta',
                ['alpha', 'beta', 'gamma', 'delta'],
                'not present',
                1.45,
                ['alpha', 'beta', 'gamma', 'delta'],
            ),
        )
        lexical_results = [
            connection.execute(
                text(
                    'SELECT score, matched_terms FROM rag_python_lexical_score_v1('
                    ':title, :searchable, :terms, :phrase)'
                ),
                {
                    'title': title,
                    'searchable': searchable,
                    'terms': terms,
                    'phrase': phrase,
                },
            ).one()
            for title, searchable, terms, phrase, _, _ in lexical_cases
        ]
        round_cases = (
            0.0,
            -0.0,
            0.0078125,
            -0.0078125,
            1.2345645,
            -1.2345645,
            0.0000005,
            -0.0000005,
            math.inf,
            -math.inf,
            math.nan,
        )
        rounded_results = [
            connection.scalar(
                text('SELECT rag_python_round6_binary64_v1(:value)'),
                {'value': value},
            )
            for value in round_cases
        ]
    assert str(rounder) == 'rag_python_round6_binary64_v1(double precision)'
    assert str(scorer) == 'rag_python_lexical_score_v1(text,text,text[],text)'
    for result, (*_, expected_score, expected_terms) in zip(
        lexical_results, lexical_cases, strict=True
    ):
        assert result.matched_terms == expected_terms
        assert struct.pack('>d', result.score) == struct.pack('>d', expected_score)
    for value, actual in zip(round_cases, rounded_results, strict=True):
        expected = round(value, 6)
        if math.isnan(expected):
            assert math.isnan(actual)
        else:
            assert struct.pack('>d', actual) == struct.pack('>d', expected)
