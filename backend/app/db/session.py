from collections.abc import Generator

from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.db.initialization import initialize_database_runtime

settings = get_settings()
_runtime = initialize_database_runtime(settings.resolved_database_url())
engine = _runtime.engine
SessionLocal = _runtime.session_factory


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
