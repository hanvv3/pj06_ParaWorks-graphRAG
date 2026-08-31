from __future__ import annotations

from dataclasses import replace

import pytest

from backend.app.agent_runtime.rag_v2_identity import (
    SecurityScope,
    ServerRagSecurityScopeResolver,
    security_scope_fingerprint,
    verify_serialized_security_scope_fingerprint,
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


def test_scope_fingerprint_binds_every_valid_mutable_scope_field() -> None:
    settings = Settings(agent_runtime_fingerprint_secret='scope-test-secret', agent_runtime_fingerprint_key_version='scope-v1')
    all_scope = _scope()
    constrained = _scope(
        resource_scope_mode='constrained',
        project_constraints=('project_key:a',),
        source_constraints=('source_pk:2',),
    )
    baseline = security_scope_fingerprint(constrained, settings=settings)

    assert len(baseline) == 64
    alternatives = (
        replace(constrained, principal_subject='user-2'),
        replace(constrained, workspace_scope_id='scope-2'),
        replace(constrained, project_constraints=('project_key:b',)),
        replace(constrained, source_constraints=('source_pk:3',)),
        replace(constrained, allowed_permission_levels=('public',)),
        all_scope,
    )
    assert all(
        baseline != security_scope_fingerprint(alternative, settings=settings)
        for alternative in alternatives
    )


@pytest.mark.parametrize(
    'field', ('auth_policy_version', 'permission_policy_version'),
)
def test_security_scope_rejects_non_frozen_policy_versions(field: str) -> None:
    with pytest.raises(ValueError):
        _scope(**{field: 'other-policy:v1'})


def test_serialized_scope_fingerprint_mismatch_stops_all_downstream_sentinels() -> None:
    settings = Settings(agent_runtime_fingerprint_secret='scope-test-secret', agent_runtime_fingerprint_key_version='scope-v1')
    scope = _scope()
    calls = {'query': 0, 'embedding': 0, 'db': 0}

    def guarded_harness() -> None:
        verify_serialized_security_scope_fingerprint(
            scope,
            serialized_fingerprint='0' * 64,
            settings=settings,
        )
        calls['query'] += 1
        calls['embedding'] += 1
        calls['db'] += 1

    with pytest.raises(ValueError):
        guarded_harness()
    assert calls == {'query': 0, 'embedding': 0, 'db': 0}
