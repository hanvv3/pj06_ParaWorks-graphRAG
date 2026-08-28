from dataclasses import dataclass

from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import has_retained_c5_state
from backend.app.core.config import Settings
from backend.app.models import (
    AgentRun,
    AssistantConversation,
    AssistantMessage,
    DecisionRecord,
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    HistoryEvent,
    ReviewItem,
    Source,
    SyncJob,
    TimelineEvent,
    Todo,
    VectorIndexState,
)

RESET_MODELS = (
    AssistantMessage,
    AssistantConversation,
    AgentRun,
    VectorIndexState,
    ReviewItem,
    DecisionRecord,
    HistoryEvent,
    TimelineEvent,
    Todo,
    DocumentParserRun,
    DocumentChunk,
    DocumentVersion,
    Document,
    Source,
    SyncJob,
)


@dataclass(frozen=True)
class DataResetResult:
    dry_run: bool
    deleted_counts: dict[str, int]
    preserved_tables: tuple[str, ...]
    retained_c5_state: bool


def reset_connector_derived_data(
    db: Session,
    *,
    settings: Settings,
    dry_run: bool = True,
    confirm: bool = False,
) -> DataResetResult:
    counts = {model.__tablename__: db.query(model).count() for model in RESET_MODELS}
    retained_c5_state = has_retained_c5_state(db)
    if dry_run:
        return _result(
            dry_run=True,
            counts=counts,
            retained_c5_state=retained_c5_state,
        )
    if settings.paraworks_env != 'local':
        raise ValueError('connector data reset is only allowed in local environment')
    if not confirm:
        raise ValueError('connector data reset requires confirm=True')
    if retained_c5_state:
        raise ValueError(
            'retained C.5 state cannot be deleted; recreate only an explicitly '
            'disposable local database'
        )

    for model in RESET_MODELS:
        db.query(model).delete(synchronize_session=False)
    db.commit()
    return _result(
        dry_run=False,
        counts=counts,
        retained_c5_state=False,
    )


def _result(
    *,
    dry_run: bool,
    counts: dict[str, int],
    retained_c5_state: bool,
) -> DataResetResult:
    return DataResetResult(
        dry_run=dry_run,
        deleted_counts=counts,
        preserved_tables=('auth_users', 'refresh_tokens', 'integration_connections', 'message_channels', 'messages'),
        retained_c5_state=retained_c5_state,
    )
