from __future__ import annotations

from dataclasses import replace

import pytest

from backend.app.agent_runtime.rag_v2_identity import (
    SecurityScope,
    ServerRagSecurityScopeResolver,
    security_scope_fingerprint,
)
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser


def _actor(*permissions: str) -> DemoUser:
    return DemoUser(
        id='user-1',
        email='user@example.test',
        role='employee',
        permission_levels=set(permissions),
        name='User',
        title='Tester',
        department='Platform',
    )


def _scope(**overrides: object) -> SecurityScope:
    values: dict[str, object] = {
        'contract_version': 'rag-security-scope:v1',
        'principal_subject': 'user-1',
        'workspace_scope_id': 'scope-1',
        'resource_scope_mode': 'all_current_scope',
        'project_constraints': (),
        'source_constraints': (),
        'allowed_permission_levels': ('public', 'internal'),
        'auth_policy_version': 'demo-auth:v1',
        'permission_policy_version': 'rag-permission-policy:v1',
    }
    values.update(overrides)
    return SecurityScope(**values)  # type: ignore[arg-type]


def test_server_scope_resolution_uses_actor_permissions_and_exact_runtime_scope() -> None:
    scope = ServerRagSecurityScopeResolver(Settings(agent_runtime_security_scope_id='workspace-a')).resolve(
        db=object(), actor=_actor('restricted', 'public')
    )

    assert scope.workspace_scope_id == 'workspace-a'
    assert scope.resource_scope_mode == 'all_current_scope'
    assert scope.project_constraints == ()
    assert scope.source_constraints == ()
    assert scope.allowed_permission_levels == ('public', 'restricted')


def test_security_scope_requires_exact_mode_namespace_order_and_narrowing() -> None:
    with pytest.raises(ValueError):
        _scope(resource_scope_mode='constrained')
    with pytest.raises(ValueError):
        _scope(project_constraints=('project_key:z', 'project_key:a'))
    with pytest.raises(ValueError):
        _scope(source_constraints=('source_pk:02',))
    with pytest.raises(ValueError):
        _scope(resource_scope_mode='all_current_scope', source_constraints=('source_pk:2',))

    constrained = _scope(
        resource_scope_mode='constrained',
        project_constraints=('project_key:a',),
        source_constraints=('source_pk:2', 'source_pk:10'),
    )
    assert constrained.source_constraints == ('source_pk:2', 'source_pk:10')


def test_scope_fingerprint_binds_every_scope_field_without_exposing_raw_principal() -> None:
    settings = Settings(agent_runtime_fingerprint_secret='scope-test-secret', agent_runtime_fingerprint_key_version='scope-v1')
    scope = _scope()
    baseline = security_scope_fingerprint(scope, settings=settings)

    assert len(baseline) == 64
    assert baseline != security_scope_fingerprint(replace(scope, workspace_scope_id='scope-2'), settings=settings)
    assert baseline != security_scope_fingerprint(replace(scope, principal_subject='user-2'), settings=settings)
