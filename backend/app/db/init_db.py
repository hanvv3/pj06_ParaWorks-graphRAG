from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

import backend.app.models  # noqa: F401
from backend.app.admin.auto_review_keys import AutoReviewKeyBootstrapService
from backend.app.core.config import get_settings
from backend.app.db.base import Base
from backend.app.db.session import engine
from backend.app.ingestion.service import ingest_events
from backend.app.seeds.auth_users import seed_auth_users
from backend.app.seeds.mock_sources import SEED_EVENTS


def init_db(engine_override: Engine | None = None) -> None:
    target_engine = engine_override or engine
    Base.metadata.create_all(bind=target_engine)
    settings = get_settings()
    if settings.paraworks_seed_demo_data:
        # Seed ingestion creates keyed state. Establish its identity first, using
        # the same fail-closed bootstrap as application startup; never adopt an
        # existing database whose keyed state has lost its identity.
        AutoReviewKeyBootstrapService(
            session_factory=sessionmaker(bind=target_engine),
            settings=settings,
        ).ensure_initialized()
    if settings.paraworks_env == 'local':
        with Session(target_engine) as db:
            seed_auth_users(db)
            db.commit()
    if settings.paraworks_seed_demo_data:
        with Session(target_engine) as db:
            ingest_events(db, SEED_EVENTS)
            db.commit()


def main() -> None:
    init_db()
    print('ParaWorks database tables are ready.')


if __name__ == '__main__':
    main()
