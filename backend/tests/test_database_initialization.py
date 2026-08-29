from __future__ import annotations

import traceback
from typing import Any

import pytest
from sqlalchemy import create_engine as sqlalchemy_create_engine
from sqlalchemy import event

from backend.app.db import initialization


def test_initialize_database_runtime_preserves_exact_options_without_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    connect_events: list[str] = []
    engine = sqlalchemy_create_engine('sqlite:///:memory:')
    event.listen(engine, 'connect', lambda *_args: connect_events.append('connect'))

    def create_engine_probe(database_url: str, **kwargs: object):
        observed['database_url'] = database_url
        observed['kwargs'] = kwargs
        return engine

    monkeypatch.setattr(initialization, 'create_engine', create_engine_probe)
    runtime = initialization.initialize_database_runtime('sqlite:///:memory:')

    assert runtime.engine is engine
    assert runtime.session_factory.kw['bind'] is engine
    assert runtime.session_factory.kw['autoflush'] is False
    assert runtime.session_factory.kw['autocommit'] is False
    assert runtime.session_factory.kw['expire_on_commit'] is True
    assert observed == {
        'database_url': 'sqlite:///:memory:',
        'kwargs': {'pool_pre_ping': True},
    }
    assert connect_events == []
    assert 'Engine(' not in repr(runtime)
    engine.dispose()


@pytest.mark.parametrize(
    'database_url',
    (
        'not-a-sqlalchemy-url-sensitive',
        'paraworks_missing_dialect://user:secret@host/database',
    ),
)
def test_configuration_failures_are_typed_and_sanitized(database_url: str) -> None:
    with pytest.raises(initialization.DatabaseConfigurationError) as captured:
        initialization.initialize_database_runtime(database_url)

    error = captured.value
    rendered = ''.join(traceback.format_exception(error))
    assert error.code == 'database_configuration_invalid'
    assert error.args == ()
    assert error.__dict__ == {}
    assert error.__cause__ is None
    assert error.__context__ is None
    assert database_url not in repr(error)
    assert database_url not in rendered
    assert 'secret' not in rendered


def test_typed_error_codes_are_class_level_only() -> None:
    configuration = initialization.DatabaseConfigurationError()
    storage = initialization.DatabaseInitializationError()

    assert configuration.code == 'database_configuration_invalid'
    assert storage.code == 'database_initialization_failed'
    assert configuration.args == storage.args == ()
    assert configuration.__dict__ == storage.__dict__ == {}
