from dataclasses import dataclass

import pytest

from backend.app.agent_runtime import (
    GraphVersionRegistry,
    RuntimeVersionUnavailable,
)


def _builder_v1(_saver: object) -> object:
    return object()


def _builder_v2(_saver: object) -> object:
    return object()


def test_registry_resolves_the_registered_workflow_version() -> None:
    registry = GraphVersionRegistry()

    registry.register(
        'company-memory-review',
        'company-memory-review-v2.0',
        _builder_v2,
    )

    assert registry.resolve(
        'company-memory-review',
        'company-memory-review-v2.0',
    ) is _builder_v2


def test_registry_rejects_replacement_of_an_existing_version() -> None:
    registry = GraphVersionRegistry()
    registry.register(
        'company-memory-review',
        'company-memory-review-v2.0',
        _builder_v2,
    )

    with pytest.raises(
        ValueError,
        match='^graph version is already registered$',
    ):
        registry.register(
            'company-memory-review',
            'company-memory-review-v2.0',
            _builder_v1,
        )

    assert registry.resolve(
        'company-memory-review',
        'company-memory-review-v2.0',
    ) is _builder_v2


@pytest.mark.parametrize(
    ('workflow_name', 'graph_version'),
    [
        ('', 'company-memory-review-v2.0'),
        ('   ', 'company-memory-review-v2.0'),
        ('company-memory-review', ''),
        ('company-memory-review', '   '),
    ],
)
def test_registry_rejects_blank_version_identity(
    workflow_name: str,
    graph_version: str,
) -> None:
    registry = GraphVersionRegistry()

    with pytest.raises(
        ValueError,
        match='^workflow_name and graph_version are required$',
    ):
        registry.register(workflow_name, graph_version, _builder_v2)


def test_registry_reports_an_unknown_version_without_fallback() -> None:
    registry = GraphVersionRegistry()
    registry.register(
        'company-memory-review',
        'company-memory-review-v1.0',
        _builder_v1,
    )

    with pytest.raises(
        RuntimeVersionUnavailable,
        match='^unsupported graph version: company-memory-review-v2.0$',
    ):
        registry.resolve(
            'company-memory-review',
            'company-memory-review-v2.0',
        )


def test_waiting_thread_uses_its_immutable_version_when_flag_is_off() -> None:
    @dataclass(frozen=True)
    class StoredThread:
        workflow_name: str
        graph_version: str

    registry = GraphVersionRegistry()
    registry.register(
        'company-memory-review',
        'company-memory-review-v1.0',
        _builder_v1,
    )
    registry.register(
        'company-memory-review',
        'company-memory-review-v2.0',
        _builder_v2,
    )
    waiting_thread = StoredThread(
        workflow_name='company-memory-review',
        graph_version='company-memory-review-v2.0',
    )
    feature_flag_enabled = False

    assert feature_flag_enabled is False
    assert registry.resolve(
        waiting_thread.workflow_name,
        waiting_thread.graph_version,
    ) is _builder_v2
