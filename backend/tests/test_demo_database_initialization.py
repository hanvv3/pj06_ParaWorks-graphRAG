import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.admin.auto_review_keys import (
    AutoReviewKeyBootstrapError,
    AutoReviewKeyBootstrapService,
)
from backend.app.core.config import Settings
from backend.app.db import init_db as initialization
from backend.app.db.base import Base
from backend.app.ingestion.service import ingest_events
from backend.app.models import Source
from backend.app.seeds.mock_sources import SEED_EVENTS


def test_seeded_demo_can_bootstrap_and_initialize_again(tmp_path, monkeypatch):
    engine = create_engine(f'sqlite:///{tmp_path / "demo.db"}')
    settings = Settings(
        _env_file=None, paraworks_demo_mode=True, paraworks_seed_demo_data=True
    )
    monkeypatch.setattr(initialization, 'get_settings', lambda: settings)
    try:
        initialization.init_db(engine)
        result = AutoReviewKeyBootstrapService(
            session_factory=sessionmaker(bind=engine),
            settings=settings,
        ).ensure_initialized()
        assert result.schema_available and not result.initialized
        assert not result.ready  # SQLite must not gain production authority.
        with Session(engine) as db:
            count = db.scalar(select(func.count()).select_from(Source))
        assert count > 0
        initialization.init_db(engine)
        with Session(engine) as db:
            assert db.scalar(select(func.count()).select_from(Source)) == count
    finally:
        engine.dispose()


def test_initializer_does_not_adopt_orphaned_keyed_demo_state(tmp_path, monkeypatch):
    engine = create_engine(f'sqlite:///{tmp_path / "orphan.db"}')
    settings = Settings(
        _env_file=None, paraworks_demo_mode=True, paraworks_seed_demo_data=True
    )
    monkeypatch.setattr(initialization, 'get_settings', lambda: settings)
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as db:
            ingest_events(db, SEED_EVENTS)
            db.commit()
            count = db.scalar(select(func.count()).select_from(Source))
        with pytest.raises(AutoReviewKeyBootstrapError):
            initialization.init_db(engine)
        with Session(engine) as db:
            assert db.scalar(select(func.count()).select_from(Source)) == count
    finally:
        engine.dispose()
