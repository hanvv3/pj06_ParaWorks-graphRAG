from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

import backend.app.models  # noqa: F401
from backend.app.db.base import Base

# Add RAG serving projection persistence.
# Revision ID: d1a2b3c4e5f6
# Revises: 9d7f3a1c6e20
# Create Date: 2026-08-31 00:00:00.000000
revision = 'd1a2b3c4e5f6'
down_revision = '9d7f3a1c6e20'
branch_labels = None
depends_on = None

_CORPUS_TABLE = 'rag_serving_corpus_generations'
_LEXICAL_TABLE = 'rag_lexical_serving_projections'
_VECTOR_TABLE = 'vector_index_states'

_VECTOR_COLUMNS = (
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
)

_VECTOR_CHECKS = (
    'ck_vector_index_state_d_provenance',
    'ck_vector_index_state_d_policy',
    'ck_vector_index_state_d_hmacs',
)


def upgrade() -> None:
    _create_model_table(_CORPUS_TABLE)
    _add_vector_columns()
    _add_vector_constraints()
    _create_model_table(_LEXICAL_TABLE)
    _install_postgresql_scorers()


def downgrade() -> None:
    _refuse_retained_serving_state()
    _drop_postgresql_scorers()
    Base.metadata.drop_all(
        op.get_bind(),
        tables=[Base.metadata.tables[_LEXICAL_TABLE]],
        checkfirst=True,
    )
    _drop_vector_constraints()
    _drop_vector_columns()
    Base.metadata.drop_all(
        op.get_bind(),
        tables=[Base.metadata.tables[_CORPUS_TABLE]],
        checkfirst=True,
    )


def _create_model_table(table_name: str) -> None:
    Base.metadata.create_all(
        op.get_bind(),
        tables=[Base.metadata.tables[table_name]],
        checkfirst=True,
    )


def _add_vector_columns() -> None:
    if _VECTOR_TABLE not in _table_names():
        return
    existing = _column_names(_VECTOR_TABLE)
    table = Base.metadata.tables[_VECTOR_TABLE]
    for column_name in _VECTOR_COLUMNS:
        if column_name not in existing:
            op.add_column(_VECTOR_TABLE, table.c[column_name]._copy())


def _add_vector_constraints() -> None:
    if _VECTOR_TABLE not in _table_names():
        return
    table = Base.metadata.tables[_VECTOR_TABLE]
    model_checks = {
        constraint.name: str(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, sa.CheckConstraint)
        and constraint.name in _VECTOR_CHECKS
    }
    for constraint_name in _VECTOR_CHECKS:
        _create_check(_VECTOR_TABLE, constraint_name, model_checks[constraint_name])
    _create_fk(
        _VECTOR_TABLE,
        'fk_vector_index_state_rag_serving_corpus',
        ['corpus_generation_id'],
        _CORPUS_TABLE,
        ['id'],
    )


def _drop_vector_constraints() -> None:
    if _VECTOR_TABLE not in _table_names():
        return
    _drop_fk(_VECTOR_TABLE, 'fk_vector_index_state_rag_serving_corpus')
    for constraint_name in reversed(_VECTOR_CHECKS):
        _drop_check(_VECTOR_TABLE, constraint_name)


def _drop_vector_columns() -> None:
    if _VECTOR_TABLE not in _table_names():
        return
    existing = _column_names(_VECTOR_TABLE)
    for column_name in reversed(_VECTOR_COLUMNS):
        if column_name in existing:
            with op.batch_alter_table(_VECTOR_TABLE) as batch:
                batch.drop_column(column_name)


def _install_postgresql_scorers() -> None:
    if op.get_bind().dialect.name != 'postgresql':
        return
    op.execute(
        sa.text(
            """
            CREATE OR REPLACE FUNCTION rag_python_round6_binary64_v1(
              value double precision
            ) RETURNS double precision
            LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE AS $$
            DECLARE
              payload bytea;
              unsigned_bits numeric := 0;
              biased_exponent integer;
              binary_exponent integer;
              significand numeric;
              numerator numeric;
              denominator numeric;
              rounded_integer numeric;
              remainder numeric;
              negative_value boolean;
              byte_ordinal integer;
            BEGIN
              payload := float8send(value);
              FOR byte_ordinal IN 0..7 LOOP
                unsigned_bits := unsigned_bits * 256 + get_byte(payload, byte_ordinal);
              END LOOP;
              negative_value := unsigned_bits >= 9223372036854775808;
              IF negative_value THEN
                unsigned_bits := unsigned_bits - 9223372036854775808;
              END IF;
              biased_exponent := floor(unsigned_bits / 4503599627370496);
              IF biased_exponent = 2047 THEN
                RAISE EXCEPTION 'non-finite lexical score';
              END IF;
              significand := mod(unsigned_bits, 4503599627370496);
              IF biased_exponent = 0 THEN
                binary_exponent := -1074;
              ELSE
                significand := significand + 4503599627370496;
                binary_exponent := biased_exponent - 1075;
              END IF;
              numerator := significand * 1000000;
              denominator := 1;
              IF binary_exponent >= 0 THEN
                numerator := numerator * power(2::numeric, binary_exponent);
              ELSE
                denominator := power(2::numeric, -binary_exponent);
              END IF;
              rounded_integer := floor(numerator / denominator);
              remainder := numerator - rounded_integer * denominator;
              IF remainder * 2 > denominator OR
                 (remainder * 2 = denominator AND mod(rounded_integer, 2) = 1) THEN
                rounded_integer := rounded_integer + 1;
              END IF;
              IF rounded_integer = 0 AND negative_value THEN
                RETURN '-0'::double precision;
              END IF;
              IF negative_value THEN
                rounded_integer := -rounded_integer;
              END IF;
              RETURN (rounded_integer / 1000000)::double precision;
            END $$
            """
        )
    )
    op.execute(
        sa.text(
            """
            CREATE OR REPLACE FUNCTION rag_python_lexical_score_v1(
              title_lower text,
              searchable_lower text,
              query_terms text[],
              phrase_lower text
            ) RETURNS TABLE(score double precision, matched_terms text[])
            LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS $$
              WITH term_rows AS (
                SELECT term, ordinal
                FROM unnest(query_terms) WITH ORDINALITY AS items(term, ordinal)
              ),
              matched AS (
                SELECT term, ordinal
                FROM term_rows
                WHERE strpos(searchable_lower, term) > 0
              ),
              aggregate_values AS (
                SELECT
                  count(*) AS matched_count,
                  count(*) FILTER (WHERE strpos(title_lower, term) > 0) AS title_hits,
                  COALESCE(
                    array_agg(term ORDER BY ordinal),
                    ARRAY[]::text[]
                  ) AS collected_terms
                FROM matched
              )
              SELECT
                CASE
                  WHEN cardinality(query_terms) = 0 OR matched_count = 0 THEN 0.0
                  ELSE rag_python_round6_binary64_v1(
                    matched_count::double precision /
                      cardinality(query_terms)::double precision +
                    CASE WHEN strpos(searchable_lower, phrase_lower) > 0
                         THEN 1.0 ELSE 0.0 END +
                    LEAST(title_hits::double precision * 0.15, 0.45)
                  )
                END AS score,
                collected_terms AS matched_terms
              FROM aggregate_values
            $$
            """
        )
    )


def _drop_postgresql_scorers() -> None:
    if op.get_bind().dialect.name != 'postgresql':
        return
    op.execute(
        sa.text(
            'DROP FUNCTION IF EXISTS '
            'rag_python_lexical_score_v1(text, text, text[], text)'
        )
    )
    op.execute(
        sa.text(
            'DROP FUNCTION IF EXISTS rag_python_round6_binary64_v1(double precision)'
        )
    )


def _refuse_retained_serving_state() -> None:
    bind = op.get_bind()
    if _CORPUS_TABLE in _table_names() and bind.scalar(
        sa.text(f'SELECT count(*) FROM {_CORPUS_TABLE}')
    ):
        raise RuntimeError(
            'refusing D schema downgrade while corpus generation state is retained'
        )
    if _LEXICAL_TABLE in _table_names() and bind.scalar(
        sa.text(f'SELECT count(*) FROM {_LEXICAL_TABLE}')
    ):
        raise RuntimeError(
            'refusing D schema downgrade while lexical projections are retained'
        )
    if (
        _VECTOR_TABLE in _table_names()
        and 'serving_kind' in _column_names(_VECTOR_TABLE)
        and bind.scalar(
            sa.text(
                f'SELECT count(*) FROM {_VECTOR_TABLE} WHERE serving_kind IS NOT NULL'
            )
        )
    ):
        raise RuntimeError(
            'refusing D schema downgrade while D vector state is retained'
        )


def _create_check(table_name: str, name: str, condition: str) -> None:
    if name in _constraint_names(table_name, 'check'):
        return
    with op.batch_alter_table(table_name) as batch:
        batch.create_check_constraint(name, condition)


def _create_fk(
    table_name: str,
    name: str,
    local_columns: list[str],
    remote_table: str,
    remote_columns: list[str],
) -> None:
    if name in _constraint_names(table_name, 'foreign_key'):
        return
    with op.batch_alter_table(table_name) as batch:
        batch.create_foreign_key(
            name,
            remote_table,
            local_columns,
            remote_columns,
            ondelete='RESTRICT',
        )


def _drop_check(table_name: str, name: str) -> None:
    if name in _constraint_names(table_name, 'check'):
        with op.batch_alter_table(table_name) as batch:
            batch.drop_constraint(name, type_='check')


def _drop_fk(table_name: str, name: str) -> None:
    if name in _constraint_names(table_name, 'foreign_key'):
        with op.batch_alter_table(table_name) as batch:
            batch.drop_constraint(name, type_='foreignkey')


def _constraint_names(table_name: str, kind: str) -> set[str | None]:
    inspector = inspect(op.get_bind())
    getters = {
        'check': inspector.get_check_constraints,
        'foreign_key': inspector.get_foreign_keys,
    }
    return {item['name'] for item in getters[kind](table_name)}


def _table_names() -> set[str]:
    return set(inspect(op.get_bind()).get_table_names())


def _column_names(table_name: str) -> set[str]:
    return {item['name'] for item in inspect(op.get_bind()).get_columns(table_name)}
