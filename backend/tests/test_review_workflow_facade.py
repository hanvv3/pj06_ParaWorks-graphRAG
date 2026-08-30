from dataclasses import dataclass, field

import pytest
from sqlalchemy.orm import Session, sessionmaker

from backend.app.agent_runtime.review_workflow_facade import (
    ReviewWorkflowFacade,
    ReviewWorkflowFacadeError,
)
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.models import AgentWorkflowThread


def _actor() -> DemoUser:
    return DemoUser(
        id='user-1',
        email='user-1@example.test',
        role='admin',
        permission_levels={'public', 'internal'},
        name='사용자',
        title='관리자',
        department='개발',
    )


@dataclass
class _FakeLifecycle:
    name: str
    calls: list[tuple[str, object]] = field(default_factory=list)

    def dry_run(self, *, actor, request):
        self.calls.append(('dry_run', request))
        return self.name

    def start(self, *, actor, request):
        self.calls.append(('start', request))
        return self.name

    def status(self, *, actor, thread_id):
        self.calls.append(('status', thread_id))
        return self.name

    def resume(self, *, actor, thread_id):
        self.calls.append(('resume', thread_id))
        return self.name

    def cancel(self, *, actor, thread_id):
        self.calls.append(('cancel', thread_id))
        return self.name


def _facade(
    db: Session,
    *,
    mode: str,
) -> tuple[ReviewWorkflowFacade, _FakeLifecycle, _FakeLifecycle]:
    v20 = _FakeLifecycle('v20')
    v21 = _FakeLifecycle('v21')
    facade = ReviewWorkflowFacade(
        session_factory=sessionmaker(
            bind=db.get_bind(), expire_on_commit=False
        ),
        settings=Settings(
            _env_file=None,
            agent_runtime_fingerprint_secret=(
                'task11-facade-secret-at-least-32-bytes'
            ),
            agent_runtime_fingerprint_key_version='task11-facade-v1',
            auto_review_mode=mode,
            auto_review_enforce_percentage=10 if mode == 'enforce' else 0,
            auto_review_validator_input_cost_per_1m_tokens=(
                '2.000000' if mode != 'disabled' else None
            ),
            auto_review_validator_output_cost_per_1m_tokens=(
                '12.000000' if mode != 'disabled' else None
            ),
            auto_review_extraction_input_cost_per_1m_tokens=(
                '0.750000' if mode != 'disabled' else None
            ),
            auto_review_extraction_output_cost_per_1m_tokens=(
                '4.500000' if mode != 'disabled' else None
            ),
        ),
        v20=v20,
        v21=v21,
    )
    return facade, v20, v21


@pytest.mark.parametrize(
    ('mode', 'expected'),
    (('disabled', 'v20'), ('shadow', 'v21'), ('enforce', 'v21')),
)
def test_new_preview_and_start_dispatch_only_from_current_mode(
    db_session: Session,
    mode: str,
    expected: str,
) -> None:
    facade, _, _ = _facade(db_session, mode=mode)

    assert facade.dry_run(actor=_actor(), request=object()) == expected
    assert facade.start(actor=_actor(), request=object()) == expected


@pytest.mark.parametrize('operation', ('status', 'resume', 'cancel'))
def test_existing_thread_dispatches_from_stored_graph_version(
    db_session: Session,
    operation: str,
) -> None:
    facade, v20, v21 = _facade(db_session, mode='disabled')
    db_session.add_all(
        [
            AgentWorkflowThread(
                thread_id='thread-v20',
                workflow_name='company-memory-review',
                graph_version='company-memory-review-v2.0',
                checkpoint_thread_id='cp-v20',
                checkpoint_store='memory',
                owner_subject_id='user-1',
                security_scope_id='default',
                input_hash='a' * 64,
                evidence_version_hash='b' * 64,
                status='created',
            ),
            AgentWorkflowThread(
                thread_id='thread-v21',
                workflow_name='company-memory-review',
                graph_version='company-memory-review-v2.1-auto-review',
                checkpoint_thread_id='cp-v21',
                checkpoint_store='memory',
                owner_subject_id='user-1',
                security_scope_id='default',
                input_hash='c' * 64,
                evidence_version_hash='d' * 64,
                status='created',
            ),
        ]
    )
    db_session.commit()

    assert getattr(facade, operation)(
        actor=_actor(), thread_id='thread-v20'
    ) == 'v20'
    assert getattr(facade, operation)(
        actor=_actor(), thread_id='thread-v21'
    ) == 'v21'
    assert v20.calls[-1] == (operation, 'thread-v20')
    assert v21.calls[-1] == (operation, 'thread-v21')


def test_unknown_or_foreign_thread_fails_closed_without_dispatch(
    db_session: Session,
) -> None:
    facade, v20, v21 = _facade(db_session, mode='shadow')
    db_session.add(
        AgentWorkflowThread(
            thread_id='thread-unknown',
            workflow_name='company-memory-review',
            graph_version='company-memory-review-v999',
            checkpoint_thread_id='cp-unknown',
            checkpoint_store='memory',
            owner_subject_id='user-1',
            security_scope_id='default',
            input_hash='e' * 64,
            evidence_version_hash='f' * 64,
            status='created',
        )
    )
    db_session.commit()

    with pytest.raises(ReviewWorkflowFacadeError, match='runtime_version_unavailable'):
        facade.status(actor=_actor(), thread_id='thread-unknown')
    assert v20.calls == []
    assert v21.calls == []
