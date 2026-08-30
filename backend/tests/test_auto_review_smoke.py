from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
from sqlalchemy import create_engine, text

from backend.app.agent_runtime.review_workflow_facade import ReviewWorkflowFacade
from backend.app.core.config import Settings
from backend.app.review.auto_review_rollout import (
    AutoReviewRolloutPolicyService,
    resolve_effective_rollout,
)


@dataclass
class _Lifecycle:
    starts: int = 0

    def start(self, **_kwargs):
        self.starts += 1
        return 'started'


def test_disabled_sqlite_smoke():
    v20 = _Lifecycle()
    v21 = _Lifecycle()
    settings = Settings(_env_file=None, auto_review_mode='disabled')
    facade = ReviewWorkflowFacade(
        session_factory=lambda: (_ for _ in ()).throw(AssertionError('db call')),
        settings=settings,
        v20=v20,
        v21=v21,
    )

    assert facade.start(actor=object(), request=object()) == 'started'
    assert v20.starts == 1
    assert v21.starts == 0


def test_postgres_control_plane_shadow_enforce_rollback_smoke():
    database_url = os.getenv('PARAWORKS_TEST_POSTGRES_URL')
    if not database_url:
        pytest.skip('PARAWORKS_TEST_POSTGRES_URL is required for release smoke')
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            assert connection.scalar(text("SELECT current_database() LIKE '%\\_test' ESCAPE '\\'")) is True
        base = AutoReviewRolloutPolicyService.default_snapshot(
            'task16-smoke-scope', 'auto-review-policy:v1'
        )
        shadow_settings = Settings(
            _env_file=None,
            agent_runtime_fingerprint_secret='task16-smoke-secret-is-at-least-32-bytes',
            auto_review_mode='shadow',
            auto_review_enforce_percentage=0,
        )
        shadow = resolve_effective_rollout(
            snapshot=base,
            stored_mode='shadow',
            stored_percentage=0,
            stored_authorization_generation=0,
            settings=shadow_settings,
        )
        assert shadow.effective_mode == 'shadow'
        enforce = resolve_effective_rollout(
            snapshot=base,
            stored_mode='enforce',
            stored_percentage=10,
            stored_authorization_generation=0,
            settings=Settings(
                _env_file=None,
                agent_runtime_fingerprint_secret=(
                    'task16-smoke-secret-is-at-least-32-bytes'
                ),
                auto_review_mode='enforce',
                auto_review_enforce_percentage=10,
            ),
        )
        assert enforce.effective_mode == 'shadow'
        rollback = resolve_effective_rollout(
            snapshot=base,
            stored_mode='enforce',
            stored_percentage=10,
            stored_authorization_generation=0,
            settings=Settings(
                _env_file=None,
                agent_runtime_fingerprint_secret=(
                    'task16-smoke-secret-is-at-least-32-bytes'
                ),
                auto_review_mode='disabled',
            ),
        )
        assert rollback.effective_mode == 'disabled'
        assert shadow_settings.auto_review_mode == 'shadow'
    finally:
        engine.dispose()
