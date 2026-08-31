from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from backend.app.agent_runtime.rag_provider_safety import (
    RagProviderSafetyError,
    RagProviderSafetyService,
)
from backend.app.db.base import Base
from backend.tests.test_rag_v2_costs import _snapshot


def test_exact_two_families_bootstrap_and_external_first_block_persists(tmp_path: Path):
    engine = create_engine('sqlite+pysqlite:///:memory:')
    Base.metadata.create_all(engine)
    connection = engine.connect()
    service = RagProviderSafetyService(
        latch_path=tmp_path / 'provider.json',
        identity_secret=b'provider-safety-test-secret',
        designated_environment_id='test',
    )
    service.bootstrap(
        connection,
        (_snapshot('query_embedding'), _snapshot('answer_generation')),
    )
    assert service.require_ready(
        connection, 'query_embedding', _snapshot('query_embedding')
    ).family_state == 'ready'
    service.block_overrun(
        connection,
        'query_embedding',
        agent_run_id=17,
        input_tokens=20,
        output_tokens=0,
        cost_usd=Decimal('0.000011'),
    )
    with pytest.raises(RagProviderSafetyError, match='blocked'):
        service.require_ready(
            connection, 'query_embedding', _snapshot('query_embedding')
        )

    restarted = RagProviderSafetyService(
        latch_path=tmp_path / 'provider.json',
        identity_secret=b'provider-safety-test-secret',
        designated_environment_id='test',
    )
    with pytest.raises(RagProviderSafetyError, match='blocked'):
        restarted.require_ready(
            connection, 'query_embedding', _snapshot('query_embedding')
        )


def test_disabled_path_validation_does_not_create_latch(tmp_path: Path):
    path = tmp_path / 'disabled.json'
    RagProviderSafetyService.validate_disabled_path(path)
    assert not path.exists()
    assert not Path(str(path) + '.lock').exists()


def test_db_rollback_cannot_clear_external_first_blocker(tmp_path: Path):
    engine = create_engine('sqlite+pysqlite:///:memory:')
    Base.metadata.create_all(engine)
    connection = engine.connect()
    service = RagProviderSafetyService(
        latch_path=tmp_path / 'provider.json',
        identity_secret=b'provider-safety-test-secret',
        designated_environment_id='test',
    )
    snapshots = (_snapshot('query_embedding'), _snapshot('answer_generation'))
    service.bootstrap(connection, snapshots)

    class FailingCommit:
        def execute(self, *args, **kwargs):
            return connection.execute(*args, **kwargs)

        def commit(self):
            raise RuntimeError('simulated DB commit failure')

        def rollback(self):
            connection.rollback()

    with pytest.raises(RagProviderSafetyError, match='persisted'):
        service.block_remediation(
            FailingCommit(),  # type: ignore[arg-type]
            'answer_generation',
            agent_run_id=18,
            input_tokens=4,
            output_tokens=2,
            cost_usd=Decimal('0.000050'),
        )
    with pytest.raises(RagProviderSafetyError, match='blocked'):
        service.require_ready(
            connection, 'answer_generation', snapshots[1]
        )
