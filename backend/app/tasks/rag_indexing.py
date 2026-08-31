from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from backend.app.core.config import Settings, get_settings
from backend.app.db.session import SessionLocal
from backend.app.models import SyncJob
from backend.app.rag.reindexing import run_rag_v2_reindex, run_reindex
from backend.app.tasks.celery_app import celery_app


@celery_app.task(name='rag.reindex')
def run_rag_reindex_job(job_id: str, dry_run: bool = True) -> dict:
    db = SessionLocal()
    try:
        return execute_rag_reindex_job(
            db=db,
            settings=get_settings(),
            job_id=job_id,
            dry_run=dry_run,
        )
    finally:
        db.close()


def enqueue_rag_reindex_job(*, job_id: str, dry_run: bool) -> None:
    run_rag_reindex_job.delay(job_id, dry_run)


@celery_app.task(name='rag.reindex_v2')
def run_rag_v2_reindex_job(
    job_id: str,
    dry_run: bool = True,
    operator_authorized: bool = False,
) -> dict:
    db = SessionLocal()
    try:
        return execute_rag_v2_reindex_job(
            db=db,
            settings=get_settings(),
            job_id=job_id,
            dry_run=dry_run,
            operator_authorized=operator_authorized,
        )
    finally:
        db.close()


def enqueue_rag_v2_reindex_job(
    *,
    job_id: str,
    dry_run: bool,
    operator_authorized: bool,
) -> None:
    run_rag_v2_reindex_job.delay(
        job_id,
        dry_run,
        operator_authorized,
    )


def execute_rag_reindex_job(
    *,
    db: Session,
    settings: Settings,
    job_id: str,
    dry_run: bool,
) -> dict:
    job = db.query(SyncJob).filter(SyncJob.job_id == job_id).one()
    job.status = 'running'
    job.message = 'indexing running'
    job.progress_pct = 20
    job.updated_at = datetime.now(UTC)
    db.commit()

    try:
        result = run_reindex(db=db, settings=settings, dry_run=dry_run)
    except Exception as exc:
        job.status = 'failed'
        job.message = f'failed: {exc}'
        job.progress_pct = 100
        job.updated_at = datetime.now(UTC)
        db.commit()
        raise

    job.status = 'complete'
    job.message = (
        f"indexed={result['indexed_count']} "
        f"skipped={result['skipped_count']} "
        f"saved_embedding_calls={result['saved_embedding_calls']}"
    )
    job.progress_pct = 100
    job.updated_at = datetime.now(UTC)
    db.commit()
    return {
        'job_id': job.job_id,
        'status': job.status,
        **result,
    }


def execute_rag_v2_reindex_job(
    *,
    db: Session,
    settings: Settings,
    job_id: str,
    dry_run: bool,
    operator_authorized: bool,
) -> dict:
    job = db.query(SyncJob).filter(SyncJob.job_id == job_id).one()
    job.status = 'running'
    job.message = 'rag v2 indexing running'
    job.progress_pct = 20
    job.updated_at = datetime.now(UTC)
    db.commit()

    try:
        result = run_rag_v2_reindex(
            db=db,
            settings=settings,
            dry_run=dry_run,
            operator_authorized=operator_authorized,
        )
    except Exception as exc:
        job.status = 'failed'
        job.message = f'failed: {exc}'
        job.progress_pct = 100
        job.updated_at = datetime.now(UTC)
        db.commit()
        raise

    job.status = 'complete'
    job.message = (
        f"indexed={result['indexed_count']} "
        f"skipped={result['skipped_count']} "
        f"tombstoned={result['tombstoned_count']} "
        f"saved_embedding_calls={result['saved_embedding_calls']} "
        f"corpus_generation={result['corpus_generation']} "
        f"vector_index_generation={result['vector_index_generation']}"
    )
    job.progress_pct = 100
    job.updated_at = datetime.now(UTC)
    db.commit()
    return {
        'job_id': job.job_id,
        'status': job.status,
        **result,
    }
