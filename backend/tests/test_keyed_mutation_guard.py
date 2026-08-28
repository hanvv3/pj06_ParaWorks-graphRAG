from sqlalchemy.dialects import postgresql

from backend.app.agent_runtime.keyed_mutation_guard import (
    AUTO_REVIEW_GLOBAL_LOCK_ORDER,
    AUTO_REVIEW_KEY_GENERATION_LOCK_ID,
    GENERATION_BARRIER_EXCLUSIVE_SQL,
    GENERATION_BARRIER_SHARED_SQL,
    TRUSTED_FINGERPRINT_PROJECTION_LOCK_ID,
    build_runtime_key_state_lock,
)


def test_fixed_advisory_lock_ids_and_global_order_are_stable() -> None:
    assert AUTO_REVIEW_KEY_GENERATION_LOCK_ID == 1066041229503628369
    assert TRUSTED_FINGERPRINT_PROJECTION_LOCK_ID == -2972884933094306491
    assert AUTO_REVIEW_GLOBAL_LOCK_ORDER == (
        'generation_barrier',
        'runtime_key_state',
        'provider_safety_states',
        'fingerprint_projection',
        'rollout_states',
        'sources',
        'workflow_threads',
        'calls_and_children',
        'review_items',
        'promotion_decisions_and_audits',
        'approval_and_evidence_links',
        'trusted_knowledge',
        'document_locks',
        'tombstones_and_vector_state',
    )


def test_generation_barrier_sql_uses_shared_and_exclusive_transaction_locks() -> None:
    assert str(GENERATION_BARRIER_SHARED_SQL) == (
        'SELECT pg_advisory_xact_lock_shared(:lock_id)'
    )
    assert str(GENERATION_BARRIER_EXCLUSIVE_SQL) == (
        'SELECT pg_advisory_xact_lock(:lock_id)'
    )


def test_runtime_key_state_lock_uses_for_share_or_for_update() -> None:
    shared = str(
        build_runtime_key_state_lock(for_update=False).compile(
            dialect=postgresql.dialect()
        )
    )
    exclusive = str(
        build_runtime_key_state_lock(for_update=True).compile(
            dialect=postgresql.dialect()
        )
    )

    assert 'FOR SHARE' in shared
    assert 'FOR UPDATE' in exclusive
