from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.core.config import get_settings
from backend.app.db.init_db import init_db
from backend.app.models import AuthUser, DocumentChunk, RefreshToken, ReviewItem, Source


def test_init_db_creates_expected_tables_on_fresh_engine() -> None:
    engine = create_engine('sqlite:///:memory:')

    init_db(engine_override=engine)

    tables = set(inspect(engine).get_table_names())
    assert {
        'sources',
        'review_items',
        'sync_jobs',
        'agent_runs',
        'vector_index_states',
        'agent_workflow_threads',
        'agent_workflow_requests',
        'agent_workflow_evidence_refs',
        'agent_runtime_schema_versions',
    } <= tables


def test_init_db_seeds_local_docker_users_without_demo_data_by_default(monkeypatch) -> None:
    monkeypatch.setenv('PARAWORKS_ENV', 'local')
    monkeypatch.delenv('PARAWORKS_SEED_DEMO_DATA', raising=False)
    get_settings.cache_clear()
    engine = create_engine(
        'sqlite://',
        connect_args={'check_same_thread': False},
        poolclass=StaticPool,
    )

    try:
        init_db(engine_override=engine)

        with Session(engine) as db:
            users_by_email = {user.email: user for user in db.query(AuthUser).all()}
            assert users_by_email['admin@paraworks.com'].role == 'admin'
            assert users_by_email['hanvv3@gmail.com'].role == 'admin'
            assert users_by_email['kjw4work@gmail.com'].role == 'admin'
            assert users_by_email['yonghee199702@gmail.com'].role == 'admin'
            assert db.query(Source).count() == 0
            assert db.query(ReviewItem).filter(ReviewItem.status == 'pending_review').count() == 0
    finally:
        get_settings.cache_clear()


def test_init_db_seeds_demo_data_only_when_smoke_seed_is_enabled(monkeypatch) -> None:
    monkeypatch.setenv('PARAWORKS_ENV', 'local')
    monkeypatch.setenv('PARAWORKS_SEED_DEMO_DATA', 'true')
    get_settings.cache_clear()
    engine = create_engine(
        'sqlite://',
        connect_args={'check_same_thread': False},
        poolclass=StaticPool,
    )

    try:
        init_db(engine_override=engine)

        with Session(engine) as db:
            assert db.query(Source).count() >= 1
            assert db.query(DocumentChunk).count() >= 1
            assert db.query(ReviewItem).filter(ReviewItem.status == 'pending_review').count() == 0
    finally:
        get_settings.cache_clear()


def test_init_db_removes_deleted_seed_accounts(monkeypatch) -> None:
    monkeypatch.setenv('PARAWORKS_ENV', 'local')
    get_settings.cache_clear()
    engine = create_engine(
        'sqlite://',
        connect_args={'check_same_thread': False},
        poolclass=StaticPool,
    )

    try:
        with Session(engine) as db:
            AuthUser.metadata.create_all(bind=engine)
            stale_user = AuthUser(
                external_id='employee-jun',
                email='jun@paraworks.com',
                display_name='Lee Jun',
                role='employee',
                department='Engineering',
                title='Backend Engineer',
                status='active',
                permission_levels=['public', 'internal'],
            )
            db.add(stale_user)
            db.flush()
            db.add(
                RefreshToken(
                    user_id=stale_user.id,
                    token_hash='stale-token-hash',
                    family_id='stale-family',
                    expires_at=stale_user.created_at,
                )
            )
            db.commit()

        init_db(engine_override=engine)

        with Session(engine) as db:
            emails = {user.email for user in db.query(AuthUser).all()}
            assert 'jun@paraworks.com' not in emails
            assert 'soyeon@paraworks.com' not in emails
            assert db.query(RefreshToken).count() == 0
    finally:
        get_settings.cache_clear()
