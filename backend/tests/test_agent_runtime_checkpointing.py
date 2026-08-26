import pytest
from psycopg.rows import dict_row

from backend.app.agent_runtime.checkpointing import (
    build_postgres_pool,
    build_strict_checkpoint_serializer,
    resolve_checkpoint_mode,
    sqlalchemy_url_to_psycopg_dsn,
)
from backend.app.agent_runtime.fingerprints import fingerprint_secret_bytes
from backend.app.agent_runtime.state import ReviewGraphState
from backend.app.core.config import Settings


def _valid_checkpoint_state() -> ReviewGraphState:
    return {
        'workflow_thread_id': 'thread-1',
        'graph_version': 'company-memory-review-v2.0',
        'input_hash': 'a' * 64,
        'evidence_version_hash': 'b' * 64,
        'review_item_ids': [1, 2],
        'review_status_counts': {'pending_review': 2},
        'phase': 'created',
        'completed_nodes': ['validate_input'],
        'error_codes': [],
    }


def test_checkpoint_mode_matrix_preserves_disabled_smoke_and_postgres_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('PARAWORKS_DEMO_MODE', raising=False)

    assert resolve_checkpoint_mode(
        Settings(_env_file=None, langgraph_review_v2_enabled=False)
    ) == 'disabled'
    assert resolve_checkpoint_mode(
        Settings(
            _env_file=None,
            paraworks_demo_mode=True,
            langgraph_review_v2_enabled=True,
        )
    ) == 'memory'
    assert resolve_checkpoint_mode(
        Settings(
            _env_file=None,
            database_url='sqlite://',
            langgraph_review_v2_enabled=True,
        )
    ) == 'memory'
    assert resolve_checkpoint_mode(
        Settings(
            _env_file=None,
            database_url='postgresql+psycopg://user:p%40ss@db/runtime',
            langgraph_review_v2_enabled=True,
        )
    ) == 'postgres'


def test_sqlalchemy_url_is_converted_to_psycopg_dsn_without_decoding_password() -> None:
    assert sqlalchemy_url_to_psycopg_dsn(
        'postgresql+psycopg://user:p%40ss@db/runtime'
    ) == 'postgresql://user:p%40ss@db/runtime'


def test_sqlite_url_rejection_does_not_echo_credentials_or_input_url() -> None:
    database_url = 'sqlite:///password-should-not-leak.db'

    with pytest.raises(ValueError) as exc_info:
        sqlalchemy_url_to_psycopg_dsn(database_url)

    error = str(exc_info.value)
    assert error == 'unsupported checkpoint database URL'
    assert 'password-should-not-leak' not in error
    assert database_url not in error


def test_postgres_pool_uses_checkpoint_safe_connection_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app.agent_runtime import checkpointing

    calls: list[dict[str, object]] = []
    expected_pool = object()

    def fake_connection_pool(*, conninfo: str, **kwargs: object) -> object:
        calls.append({'conninfo': conninfo, **kwargs})
        return expected_pool

    monkeypatch.setattr(checkpointing, 'ConnectionPool', fake_connection_pool)

    pool = build_postgres_pool('postgresql://user:secret@db/runtime')

    assert pool is expected_pool
    assert calls == [
        {
            'conninfo': 'postgresql://user:secret@db/runtime',
            'kwargs': {'autocommit': True, 'row_factory': dict_row},
            'min_size': 1,
            'max_size': 4,
            'open': False,
        }
    ]


def test_strict_serializer_round_trips_only_json_safe_checkpoint_state() -> None:
    serializer = build_strict_checkpoint_serializer()
    state = _valid_checkpoint_state()

    serialized = serializer.dumps_typed(state)

    assert serializer.pickle_fallback is False
    assert serializer.loads_typed(serialized) == state


def test_strict_serializer_rejects_arbitrary_python_objects() -> None:
    serializer = build_strict_checkpoint_serializer()

    with pytest.raises(TypeError, match='not msgpack serializable'):
        serializer.dumps_typed(object())


def test_production_rejects_the_local_default_fingerprint_secret() -> None:
    settings = Settings(_env_file=None, paraworks_env='production')

    with pytest.raises(ValueError, match='dedicated fingerprint secret'):
        fingerprint_secret_bytes(settings)


def test_fingerprint_configuration_rejects_an_empty_key_version() -> None:
    settings = Settings(
        _env_file=None,
        agent_runtime_fingerprint_secret='production-secret',
        agent_runtime_fingerprint_key_version=' ',
    )

    with pytest.raises(ValueError, match='configuration is incomplete'):
        fingerprint_secret_bytes(settings)


def test_production_fingerprint_configuration_returns_secret_and_key_version() -> None:
    settings = Settings(
        _env_file=None,
        paraworks_env='production',
        agent_runtime_fingerprint_secret='production-secret',
        agent_runtime_fingerprint_key_version='key-v2',
    )

    assert fingerprint_secret_bytes(settings) == (b'production-secret', 'key-v2')
