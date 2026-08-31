from __future__ import annotations

import importlib.util
import os
import struct
from collections.abc import Iterator
from pathlib import Path
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


def test_sqlite_upgrade_creates_additive_serving_schema(sqlite_migration) -> None:
    config, database_url = sqlite_migration
    command.upgrade(config, 'head')
    engine = create_engine(database_url)
    schema = inspect(engine)

    assert {'rag_serving_corpus_generations', 'rag_lexical_serving_projections'} <= set(
        schema.get_table_names()
    )
    assert {
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
    } <= {column['name'] for column in schema.get_columns('vector_index_states')}
    with engine.connect() as connection:
        assert (
            connection.scalar(text('SELECT version_num FROM alembic_version'))
            == REVISION
        )


def test_sqlite_upgrade_preserves_historical_vector_without_d_retagging(
    sqlite_migration,
) -> None:
    config, database_url = sqlite_migration
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
        connection.execute(
            text(
                'INSERT INTO vector_index_states VALUES '
                "(1, 'legacy:1', 'legacy:8', 8, :content_hash, 'indexed', NULL, CURRENT_TIMESTAMP)"
            ),
            {'content_hash': '0' * 64},
        )
    command.stamp(config, PREVIOUS_REVISION)

    command.upgrade(config, 'head')

    with engine.connect() as connection:
        row = connection.execute(
            text(
                'SELECT serving_kind, serving_identity_hmac, index_policy_version, '
                'vector_index_generation, vector_index_state_hmac '
                'FROM vector_index_states WHERE id=1'
            )
        ).one()
    assert tuple(row) == (None, None, None, None, None)


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
        result = connection.execute(
            text(
                'SELECT score, matched_terms FROM rag_python_lexical_score_v1('
                ":title, :searchable, ARRAY['redis','redis','queue','absent'], :phrase)"
            ),
            {
                'title': 'redis queue migration',
                'searchable': 'redis queue migration\nredis remains searchable',
                'phrase': 'redis queue',
            },
        ).one()
        rounded = connection.scalar(
            text('SELECT rag_python_round6_binary64_v1(:value)'),
            {'value': 1.2345645},
        )
    assert str(rounder) == 'rag_python_round6_binary64_v1(double precision)'
    assert str(scorer) == 'rag_python_lexical_score_v1(text,text,text[],text)'
    assert result.matched_terms == ['redis', 'redis', 'queue']
    assert struct.pack('>d', result.score) == struct.pack(
        '>d', round(3 / 4 + 1.0 + 0.45, 6)
    )
    assert struct.pack('>d', rounded) == struct.pack('>d', round(1.2345645, 6))
