from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.admin.auto_review_keys import (
    AutoReviewKeyBootstrapError,
    AutoReviewKeyBootstrapService,
)
from backend.app.core.config import Settings
from backend.app.db.base import Base
from backend.app.models import (
    AgentWorkflowRequest,
    AutoReviewRuntimeKeyState,
    TrustedKnowledgeFingerprintProjectionState,
)


def _service(engine, *, settings: Settings | None = None):
    return AutoReviewKeyBootstrapService(
        session_factory=sessionmaker(bind=engine, expire_on_commit=False),
        settings=settings or Settings(_env_file=None, paraworks_demo_mode=True),
    )


def test_bootstrap_is_a_noop_before_c5_tables_exist() -> None:
    engine = create_engine('sqlite:///:memory:')

    result = _service(engine).ensure_initialized()

    assert result.schema_available is False
    assert result.initialized is False


def test_disabled_sqlite_bootstrap_initializes_not_ready_state_idempotently() -> None:
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    service = _service(engine)

    first = service.ensure_initialized()
    second = service.ensure_initialized()

    assert first.schema_available is True
    assert first.initialized is True
    assert first.ready is False
    assert second.initialized is False
    assert second.ready is False
    with sessionmaker(bind=engine)() as db:
        runtime = db.query(AutoReviewRuntimeKeyState).one()
        projection = db.query(TrustedKnowledgeFingerprintProjectionState).one()
        assert runtime.component == 'auto_review_trust_promotion'
        assert runtime.generation == 1
        assert runtime.ready is False
        assert projection.ready is False
        assert projection.rebuild_required is True


def test_bootstrap_refuses_missing_runtime_state_with_retained_keyed_artifact() -> None:
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as db:
        db.add(
            AgentWorkflowRequest(
                workflow_thread_id='retained-c5-request',
                input_schema_version='review-workflow:v2.1',
                request_kind='review_source_versions',
                agent_names=['timeline_agent'],
                selection_policy_version='company-memory-review-selection:v1',
                input_hash='a' * 64,
                fingerprint_key_version='v1',
                fingerprint_key_material_verifier='b' * 64,
                auto_review_mode='shadow',
            )
        )
        db.commit()

    try:
        _service(engine).ensure_initialized()
    except AutoReviewKeyBootstrapError as exc:
        assert exc.code == 'retained_keyed_state_without_runtime_identity'
    else:
        raise AssertionError('retained keyed state must refuse bootstrap repair')


def test_bootstrap_refuses_runtime_key_identity_mismatch() -> None:
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    settings = Settings(
        _env_file=None,
        paraworks_demo_mode=True,
        agent_runtime_fingerprint_secret='x' * 32,
        agent_runtime_fingerprint_key_version='v1',
    )
    service = _service(engine, settings=settings)
    service.ensure_initialized()
    with sessionmaker(bind=engine)() as db:
        row = db.query(AutoReviewRuntimeKeyState).one()
        row.fingerprint_key_version = 'different'
        db.commit()

    try:
        service.ensure_initialized()
    except AutoReviewKeyBootstrapError as exc:
        assert exc.code == 'runtime_key_identity_mismatch'
    else:
        raise AssertionError('runtime key identity mismatch must fail closed')
