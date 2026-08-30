import json
from pathlib import Path

from backend.app.review.auto_review_evaluation import evaluate_fixture

FIXTURE = Path(__file__).parent / 'fixtures' / 'auto_review_golden_v1.json'


def test_korean_first_golden_fixture_covers_frozen_risk_categories():
    fixture = json.loads(FIXTURE.read_text(encoding='utf-8'))
    categories = {case['category'] for case in fixture['cases']}
    assert categories == {
        'direct_timeline', 'direct_history', 'proposal_not_decision',
        'planned_not_completed', 'affirmation_not_negation',
        'conditional_uncertain', 'date_subject_actor_mismatch',
        'conflicting_sources', 'source_supersession', 'permission_loss_unknown',
        'partial_support', 'high_confidence_hard_negative', 'prompt_injection',
        'exact_reaffirmation', 'trusted_collision', 'decision', 'todo', 'restricted',
    }


def test_golden_release_metrics_meet_precision_and_zero_prohibited_counts():
    report = evaluate_fixture(FIXTURE)
    assert report['gate_passed'] is True
    assert report['metrics']['auto_approval_precision'] >= 0.99
    for key in (
        'hard_negative_false_approval_count',
        'permission_or_version_violation_count',
        'duplicate_promotion_count',
        'cross_item_revoke_count',
        'malformed_output_approval_count',
        'validation_replay_mismatch_count',
    ):
        assert report['metrics'][key] == 0
