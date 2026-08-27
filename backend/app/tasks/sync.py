from logging import getLogger

from backend.app.connectors.factory import (
    ConnectorNotConfiguredError,
    get_sync_connector,
)
from backend.app.core.config import get_settings
from backend.app.db.session import SessionLocal
from backend.app.ingestion.sync import sync_connector_events
from backend.app.tasks.celery_app import celery_app

logger = getLogger(__name__)


@celery_app.task(name='sync.google_drive')
def sync_google_drive_task() -> dict[str, object]:
    settings = get_settings()
    if not settings.google_drive_sync_enabled:
        logger.info('Google Drive sync is disabled.')
        return {'status': 'disabled'}

    db = SessionLocal()
    try:
        logger.info('Starting periodic Google Drive sync...')
        # In demo mode, get_sync_connector will return a mock connector.
        # In live mode, it will look for an installed connection.
        connector = get_sync_connector('drive', settings, db=db)
        result = sync_connector_events(db=db, connector=connector)
        
        logger.info(
            f'Periodic Google Drive sync complete: {result.status} '
            f'(fetched={result.fetched_events}, created={result.created_review_items}, skipped={result.skipped_events})'
        )
        return {
            'status': result.status,
            'job_id': result.job_id,
            'fetched_events': result.fetched_events,
            'created_review_items': result.created_review_items,
            'skipped_events': result.skipped_events,
        }
    except ConnectorNotConfiguredError:
        logger.info('Google Drive connector not configured/connected. Skipping periodic sync.')
        return {'status': 'not_configured'}
    except Exception as exc:
        logger.error(f'Periodic Google Drive sync failed: {exc}', exc_info=True)
        return {'status': 'failed', 'error': str(exc)}
    finally:
        db.close()


@celery_app.task(name='sync.gmail')
def sync_gmail_task() -> dict[str, object]:
    settings = get_settings()
    if not settings.gmail_sync_enabled:
        logger.info('Gmail sync is disabled.')
        return {'status': 'disabled'}

    db = SessionLocal()
    try:
        logger.info('Starting periodic Gmail sync...')
        connector = get_sync_connector('gmail', settings, db=db)
        result = sync_connector_events(db=db, connector=connector)

        logger.info(
            f'Periodic Gmail sync complete: {result.status} '
            f'(fetched={result.fetched_events}, created={result.created_review_items}, skipped={result.skipped_events})'
        )
        return {
            'status': result.status,
            'job_id': result.job_id,
            'fetched_events': result.fetched_events,
            'created_review_items': result.created_review_items,
            'skipped_events': result.skipped_events,
        }
    except ConnectorNotConfiguredError:
        logger.info('Gmail connector not configured/connected. Skipping periodic sync.')
        return {'status': 'not_configured'}
    except Exception as exc:
        logger.error(f'Periodic Gmail sync failed: {exc}', exc_info=True)
        return {'status': 'failed', 'error': str(exc)}
    finally:
        db.close()

