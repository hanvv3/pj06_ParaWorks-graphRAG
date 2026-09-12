from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / 'fixtures' / 'rag_v2_provider_free_golden_60.json'


def _cases():
    return json.loads(FIXTURE.read_text(encoding='utf-8'))['cases']


def test_provider_free_golden_is_exact_unique_60_case_manifest():
    """Catches silently dropping a rollout cohort or replacing fake transports."""
    cases = _cases()
    assert len(cases) == 60
    assert len({case['case_id'] for case in cases}) == 60
    assert {case['surface'] for case in cases} == {'ask', 'search', 'assistant'}
    assert {case['backend'] for case in cases} == {'keyword', 'pgvector'}
    assert {case['evidence_kind'] for case in cases} == {
        'trusted', 'raw', 'no_match', 'hidden', 'stale', 'revoked',
        'hard_negative', 'v1_parity',
    }
    assert {case['permission'] for case in cases} == {'public', 'internal', 'restricted'}


@pytest.mark.parametrize('case', _cases(), ids=lambda case: case['case_id'])
def test_provider_free_golden_has_zero_external_dispatch_or_leak(case):
    """Catches network/provider use or permission/stale/revoke/evidence leakage."""
    from backend.app.agent_runtime.rag_rollout import RagRolloutPolicy

    plan = RagRolloutPolicy().plan(
        mode='shadow', stage='assistant', surface=case['surface'], backend=case['backend'],
    )
    assert plan.v2_generation_count == 0
    assert case['model_transport'] == 'fake'
    assert case['embedding_transport'] == 'fake'
    assert case['network_calls'] == 0
    assert case['provider_calls'] == 0
    assert case['permission_leak_count'] == 0
    assert case['revoked_leak_count'] == 0
    assert case['stale_leak_count'] == 0
    assert case['invalid_evidence_slot_count'] == 0
    assert case['missing_evidence_slot_count'] == 0
