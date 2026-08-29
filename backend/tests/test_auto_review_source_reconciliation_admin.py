from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from backend.app.admin.auto_review_source_reconciliation import (
    command_exit_code,
    format_aggregate_result,
)
from backend.app.core.config import Settings
from backend.app.rag.pgvector_store import PgVectorStore
from backend.app.review.auto_review_source_reconciliation import (
    SourceReconciliationResult,
)


def test_source_reconciliation_admin_has_fixed_actor_aggregate_output_and_exit_codes() -> None:
    clean = SourceReconciliationResult(readiness=True)
    retained = SourceReconciliationResult(
        stale_count=1,
        remaining_count=1,
        readiness=False,
    )

    output = format_aggregate_result(retained)

    assert 'system:local-auto-review-source-reconciler' in output
    assert 'stale=1' in output
    assert 'remaining=1' in output
    assert 'source_id' not in output
    assert command_exit_code(clean) == 0
    assert command_exit_code(retained) == 3


def test_source_reconciliation_admin_bounds_database_failures(
    monkeypatch, capsys
) -> None:
    import backend.app.admin.auto_review_source_reconciliation as admin

    class FailingSession:
        def __enter__(self):
            raise OperationalError('SELECT secret', {}, RuntimeError('boom'))

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(admin, 'SessionLocal', lambda: FailingSession())

    assert admin.main(['status', '--limit', '1']) == 3
    output = capsys.readouterr().out
    assert 'failures=1' in output
    assert 'readiness=false' in output
    assert 'SELECT secret' not in output


def test_reconciliation_factory_injects_transaction_bound_pgvector_only_for_postgres() -> None:
    from backend.app.review.auto_review_source_reconciliation import (
        build_source_reconciliation_service,
    )

    class _Dialect:
        name = 'postgresql'

    class _Bind:
        dialect = _Dialect()

    class _PostgresSession:
        def get_bind(self):
            return _Bind()

    postgres_db = _PostgresSession()
    settings = Settings(
        database_url='postgresql+psycopg://role_test:unused@localhost/db_test',
        openai_embedding_dimensions=8,
    )

    postgres_service = build_source_reconciliation_service(
        postgres_db,  # type: ignore[arg-type]
        settings=settings,
    )

    assert isinstance(postgres_service._vector_writer, PgVectorStore)
    assert postgres_service._vector_writer.session is postgres_db
    assert postgres_service._vector_writer.config.embedding_dimensions == 8

    engine = create_engine('sqlite://')
    try:
        with sessionmaker(bind=engine)() as sqlite_db:
            sqlite_service = build_source_reconciliation_service(
                sqlite_db,
                settings=Settings(database_url='sqlite://'),
            )
            assert sqlite_service._vector_writer is None
    finally:
        engine.dispose()
