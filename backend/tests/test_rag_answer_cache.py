import importlib
import json
import os
from dataclasses import replace
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, inspect, text

from backend.app.agent_runtime.fingerprints import keyed_fingerprint
from backend.app.agent_runtime.rag_v2_identity import security_scope_fingerprint
from backend.app.agents.rag_orchestrator_agent.v2_answer_schema import (
    RagAnswerOutputValidator,
)
from backend.app.rag.evidence_projection import (
    CanonicalEvidenceProjector,
    _build_prepared_model_influence_set_hmac_v2,
)
from backend.app.rag.retrieval import EvidenceSlot
from backend.tests.graph_projection_fixtures import SCOPE, SETTINGS, seed_corpus
from backend.tests.postgres_isolation import lease_postgres_schema
from backend.tests.test_neo4j_retriever import adapter, paths, request


def modules():
    assert importlib.util.find_spec('backend.app.rag.answer_cache'), (
        'answer cache contract missing'
    )
    return (
        importlib.import_module('backend.app.rag.answer_cache'),
        importlib.import_module('backend.app.rag.answer_cache_store'),
    )


@pytest.fixture
def prepared(db_session):
    seed_corpus(db_session)
    expected = paths(db_session)

    class Graph:
        def traverse(self, **kwargs):
            return expected

    result = adapter(db_session, Graph()).invoke(request())
    slots = tuple(
        EvidenceSlot(
            f'E{i}',
            c.evidence.support_mode,
            c.evidence,
            c.relevance_score,
            c.matched_terms,
        )
        for i, c in enumerate(result.visible, 1)
    )
    influence = CanonicalEvidenceProjector(
        db=db_session, settings=SETTINGS
    ).prepare_model_influence(
        slots,
        scope=SCOPE,
        prepared_corpus_generation=1,
        prepared_index_generation=None,
        prepared_readiness_hmac=None,
        rendered_input_hmac='a' * 64,
        graph_paths=result.graph_paths,
    )
    assert len(slots) >= 2 and influence.graph_paths
    return slots, influence


@pytest.fixture
def validator():
    return RagAnswerOutputValidator(
        signer=lambda kind, payload: keyed_fingerprint(
            payload,
            secret=b'fake-test-answer-signing-key',
            schema_version=kind,
            policy_version='test:v1',
        )
    )


def answer(slots, validator):
    return validator.validate(
        {
            'answer_blocks': [
                {
                    'text': '검증된 답변입니다.',
                    'evidence_slot_ids': ['E1'],
                    'support_mode': slots[0].support_mode,
                }
            ],
            'insufficient_evidence_reason': None,
        },
        slots=slots,
    )


def key(influence, **changes):
    contract, _ = modules()
    values = {
        'scope': SCOPE,
        'prepared': influence,
        'settings': SETTINGS,
        'versions': contract.AnswerCacheVersions(
            prompt_hmac='b' * 64,
            model_hmac='c' * 64,
            output_hmac='d' * 64,
            policy_hmac='e' * 64,
            retrieval_policy='retrieval:v1',
            graph_policy='graph:v1',
            seed_backend='keyword',
            effective_backend='neo4j',
        ),
    }
    values.update(changes)
    return contract.build_answer_cache_key(**values)


def authenticated(influence, *, scope=SCOPE, **changes):
    """Simulate a new trusted preparation, not mutation of an old signature."""
    influence = replace(influence, **changes)
    scope_hmac = security_scope_fingerprint(scope, settings=SETTINGS)
    graph_paths = tuple(
        replace(
            path,
            nodes=tuple(replace(node, scope_id=scope_hmac) for node in path.nodes),
            edges=tuple(replace(edge, scope_id=scope_hmac) for edge in path.edges),
        )
        for path in influence.graph_paths
    )
    influence = replace(influence, graph_paths=graph_paths, graph_scope=scope)
    return replace(
        influence,
        aggregate_observation_hmac=_build_prepared_model_influence_set_hmac_v2(
            observations=influence.observations,
            prepared_corpus_generation=influence.prepared_corpus_generation,
            prepared_index_generation=influence.prepared_index_generation,
            prepared_readiness_hmac=influence.prepared_readiness_hmac,
            rendered_input_hmac=influence.rendered_input_hmac,
            graph_paths=graph_paths,
            settings=SETTINGS,
        ),
    )


def test_key_binds_exact_scope_all_versions_and_full_dependencies(prepared):
    slots, influence = prepared
    base = key(influence)
    assert base == key(influence)
    for scope in (
        replace(SCOPE, principal_subject='same-role-other-user'),
        replace(SCOPE, workspace_scope_id='other-workspace'),
        replace(SCOPE, allowed_permission_levels=('public',)),
    ):
        assert (
            key(authenticated(influence, scope=scope), scope=scope).key_hmac
            != base.key_hmac
        )
    for name, value in (
        ('prompt_hmac', 'f' * 64),
        ('model_hmac', 'f' * 64),
        ('output_hmac', 'f' * 64),
        ('policy_hmac', 'f' * 64),
        ('retrieval_policy', 'retrieval:v2'),
        ('graph_policy', 'graph:off'),
        ('seed_backend', 'pgvector'),
        ('effective_backend', 'keyword'),
    ):
        assert (
            key(influence, versions=replace(base.versions, **{name: value})).key_hmac
            != base.key_hmac
        )
    assert (
        key(
            influence,
            settings=SETTINGS.model_copy(
                update={'agent_runtime_fingerprint_key_version': 'rotated'}
            ),
        ).key_hmac
        != base.key_hmac
    )
    # Authenticated prepared carriers must be rebuilt when any content changes.
    for changed in (
        replace(influence, rendered_input_hmac='f' * 64),
        replace(influence, observations=influence.observations[:1]),
        replace(influence, observations=tuple(reversed(influence.observations))),
        replace(
            influence,
            graph_paths=(
                replace(
                    influence.graph_paths[0],
                    edges=(
                        replace(influence.graph_paths[0].edges[0], version='f' * 64),
                    ),
                ),
            ),
        ),
    ):
        with pytest.raises(ValueError):
            key(changed)
    assert (
        key(authenticated(influence, rendered_input_hmac='f' * 64)).key_hmac
        != base.key_hmac
    )
    assert (
        key(authenticated(influence, observations=influence.observations[:1])).key_hmac
        != base.key_hmac
    )
    relation = replace(
        influence.graph_paths[0],
        edges=(replace(influence.graph_paths[0].edges[0], version='f' * 64),),
    )
    assert (
        key(authenticated(influence, graph_paths=(relation,))).key_hmac != base.key_hmac
    )


@pytest.fixture
def pg_engine():
    url = os.getenv('PARAWORKS_TEST_POSTGRES_URL')
    if not url:
        pytest.skip('disposable PostgreSQL required')
    with lease_postgres_schema(
        url, run_id=uuid4().hex[:12], scope_name='c1_cache'
    ) as lease:
        engine = create_engine(lease.database_url)
        try:
            yield engine
        finally:
            engine.dispose()


def migrate(engine, direction='upgrade'):
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    migration = importlib.import_module(
        'backend.migrations.versions.a6b7c8d9e0f1_add_answer_cache'
    )
    with engine.begin() as conn, Operations.context(MigrationContext.configure(conn)):
        getattr(migration, direction)()


def store(engine, validator, clock, *, enabled=True):
    _, stores = modules()
    return stores.create_answer_cache(
        engine=engine,
        settings=SETTINGS,
        validator=validator,
        enabled=enabled,
        clock=lambda: clock[0],
    )


def test_pg_hit_is_authenticated_scoped_and_non_sliding_then_bounded_cleanup(
    pg_engine, prepared, validator
):
    slots, influence = prepared
    k = key(influence)
    migrate(pg_engine)
    now = [1_800_000_000]
    cache = store(pg_engine, validator, now)
    valid = answer(slots, validator)
    contract, _ = modules()
    contract.validate_cache_key(k, slots=slots, settings=SETTINGS)
    assert contract.eligible_answer_payload(valid, slots=slots, validator=validator)
    assert cache.put(k, answer=valid, slots=slots)
    hit = cache.get(k, slots=slots)
    assert hit.answer == valid
    assert hit.key_hmac == k.key_hmac and hit.scope_hmac == k.scope_hmac
    assert len(hit.value_hmac) == 64 and hit.created_at == now[0]
    assert hit.expires_at == now[0] + 3600
    other_scope = replace(SCOPE, principal_subject='other')
    foreign = key(authenticated(influence, scope=other_scope), scope=other_scope)
    assert cache.get(foreign, slots=slots) is None
    for name in (
        'answer_json',
        'dependencies_json',
        'value_hmac',
        'scope_hmac',
        'created_at',
        'expires_at',
    ):
        with pg_engine.begin() as conn:
            row = (
                conn.execute(text('SELECT * FROM rag_answer_cache_entries'))
                .mappings()
                .one()
            )
            original = row[name]
            altered = (
                original + 1
                if name in {'created_at', 'expires_at'}
                else (
                    'f' * 64 if name in {'value_hmac', 'scope_hmac'} else original + 'x'
                )
            )
            conn.execute(
                text(f'UPDATE rag_answer_cache_entries SET {name}=:value'),
                {'value': altered},
            )
        assert cache.get(k, slots=slots) is None
        with pg_engine.begin() as conn:
            conn.execute(
                text(f'UPDATE rag_answer_cache_entries SET {name}=:value'),
                {'value': original},
            )
    changed_context = key(authenticated(influence, rendered_input_hmac='f' * 64))
    assert cache.get(changed_context, slots=slots) is None
    cache.settings = SETTINGS.model_copy(
        update={'agent_runtime_fingerprint_secret': 'rotated-cache-key-material'}
    )
    assert cache.get(k, slots=slots) is None
    cache.settings = SETTINGS
    now[0] += 3599
    assert cache.get(k, slots=slots).answer == valid
    assert cache.put(k, answer=valid, slots=slots) is False  # no sliding refresh
    now[0] += 1
    assert cache.get(k, slots=slots) is None
    assert cache.cleanup(limit=1) == 1
    assert cache.cleanup(limit=1) == 0
    migrate(pg_engine, 'downgrade')
    assert 'rag_answer_cache_entries' not in inspect(pg_engine).get_table_names()
    migrate(pg_engine)
    assert cache.put(k, answer=valid, slots=slots)


def test_only_validated_substantive_blocks_and_reference_fields_are_stored(
    pg_engine, prepared, validator
):
    slots, influence = prepared
    k = key(influence)
    migrate(pg_engine)
    cache = store(pg_engine, validator, [1_800_000_000])
    valid = answer(slots, validator)
    empty = validator.validate(
        {'answer_blocks': [], 'insufficient_evidence_reason': 'no match'}, slots=slots
    )
    for invalid in (
        empty,
        {'answer_blocks': []},
        replace(valid, assembled_answer='forged'),
        replace(valid, blocks=(replace(valid.blocks[0], text='forged'),)),
    ):
        assert cache.put(k, answer=invalid, slots=slots) is False
    assert cache.put(k, answer=valid, slots=slots)
    with pg_engine.connect() as conn:
        row = (
            conn.execute(text('SELECT * FROM rag_answer_cache_entries'))
            .mappings()
            .one()
        )
    fields = set(json.loads(row['answer_json']))
    assert fields == {'answer_blocks', 'insufficient_evidence_reason'}
    raw = row['dependencies_json']
    for forbidden in (
        'question',
        'prompt',
        'source_url',
        'source_snippet',
        'model_content"',
        'receipt',
        'token',
        'charged_cost',
        'https://',
        'Canonical approved knowledge',
    ):
        assert forbidden not in raw
    deps = json.loads(raw)
    assert len(deps['observations']) == len(slots)
    assert deps['graph_paths']


def test_default_off_sqlite_null_and_store_faults_are_safe_misses(
    pg_engine, prepared, validator
):
    slots, influence = prepared
    k = key(influence)
    valid = answer(slots, validator)
    sqlite = create_engine('sqlite://')
    try:
        for cache in (
            store(sqlite, validator, [0]),
            store(pg_engine, validator, [0], enabled=False),
            store(pg_engine, validator, [1_800_000_000]),
        ):
            assert cache.put(k, answer=valid, slots=slots) is False
            assert cache.get(k, slots=slots) is None
            assert cache.cleanup(limit=1) == 0
        assert inspect(sqlite).get_table_names() == []
    finally:
        sqlite.dispose()


def test_full_migration_chain_existing_and_fresh_pg(pg_engine, monkeypatch):
    from alembic import command
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    from backend.app.core.config import get_settings

    config = Config('alembic.ini')
    assert ScriptDirectory.from_config(config).get_heads() == ['a6b7c8d9e0f1']
    monkeypatch.setenv('PARAWORKS_DEMO_MODE', 'false')
    monkeypatch.setenv(
        'PARAWORKS_DATABASE_URL', pg_engine.url.render_as_string(hide_password=False)
    )
    get_settings.cache_clear()
    assert get_settings().resolved_database_url() == pg_engine.url.render_as_string(
        hide_password=False
    )
    try:
        command.upgrade(config, 'head')
        assert 'rag_answer_cache_entries' in inspect(pg_engine).get_table_names()
        command.downgrade(config, 'd7a8b9c0d1e2')
        assert 'rag_answer_cache_entries' not in inspect(pg_engine).get_table_names()
        assert 'sources' in inspect(pg_engine).get_table_names()
        command.upgrade(config, 'head')
        assert 'rag_answer_cache_entries' in inspect(pg_engine).get_table_names()
    finally:
        get_settings.cache_clear()


def test_cleanup_bound_future_timestamp_and_database_retention_cap(
    pg_engine, prepared, validator
):
    from sqlalchemy.exc import IntegrityError

    slots, influence = prepared
    base = key(influence)
    migrate(pg_engine)
    now = [1_800_000_000]
    cache = store(pg_engine, validator, now)
    valid = answer(slots, validator)
    keys = [
        key(influence, versions=replace(base.versions, retrieval_policy=f'v{i}'))
        for i in range(3)
    ]
    for k in keys:
        assert cache.put(k, answer=valid, slots=slots)
    now[0] -= 1
    assert cache.get(keys[0], slots=slots) is None
    now[0] += 3601
    assert cache.cleanup(limit=1) == 1
    with pg_engine.connect() as conn:
        assert conn.scalar(text('SELECT count(*) FROM rag_answer_cache_entries')) == 2
    for invalid in (0, -1, 1001, True):
        with pytest.raises(ValueError):
            cache.cleanup(limit=invalid)
    with pytest.raises(IntegrityError), pg_engine.begin() as conn:
        conn.execute(
            text('UPDATE rag_answer_cache_entries SET expires_at = created_at + 86401')
        )
    assert cache.cleanup(limit=100) == 2


def test_expiry_during_storage_read_is_a_miss(pg_engine, prepared, validator):
    slots, influence = prepared
    k = key(influence)
    migrate(pg_engine)
    cache = store(pg_engine, validator, [1_800_000_000])
    assert cache.put(k, answer=answer(slots, validator), slots=slots)
    ticks = iter((1_800_003_599, 1_800_003_600))
    cache.clock = lambda: next(ticks)
    assert cache.get(k, slots=slots) is None


def test_null_never_connects_and_pg_connection_failure_is_safe(prepared, validator):
    from sqlalchemy import event

    slots, influence = prepared
    k = key(influence)
    valid = answer(slots, validator)
    _, stores = modules()
    sqlite = create_engine('sqlite://')
    unavailable = create_engine(
        'postgresql+psycopg://unused:unused@127.0.0.1:1/unused?connect_timeout=1'
    )

    def forbid(*args, **kwargs):
        pytest.fail('Null cache must not open a connection')

    try:
        for engine in (sqlite, unavailable):
            event.listen(engine, 'do_connect', forbid)
            null = stores.create_answer_cache(
                engine=engine, settings=SETTINGS, validator=validator
            )
            assert null.get(k, slots=slots) is None
            assert not null.put(k, answer=valid, slots=slots)
            assert null.cleanup() == 0
            if engine is sqlite:
                null = store(engine, validator, [0], enabled=True)
                assert not null.put(k, answer=valid, slots=slots)
            event.remove(engine, 'do_connect', forbid)
        cache = store(unavailable, validator, [1_800_000_000])
        assert cache.get(k, slots=slots) is None
        assert not cache.put(k, answer=valid, slots=slots)
        assert cache.cleanup() == 0
        assert inspect(sqlite).get_table_names() == []
    finally:
        sqlite.dispose()
        unavailable.dispose()
