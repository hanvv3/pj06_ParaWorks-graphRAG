import json
from pathlib import Path

from langchain_core.messages import AIMessage

from backend.app.review.auto_review_evaluation import main

FIXTURE = Path(__file__).parent / 'fixtures' / 'auto_review_golden_v1.json'


def test_fake_cli_emits_only_aggregate_identity_metrics(capsys):
    code = main(['--fixture', str(FIXTURE), '--mode', 'fake', '--aggregate-only'])
    assert code == 0
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert report['safety_key'] == {
        'purpose': 'validation', 'provider': 'openai',
        'model': 'gpt-5.6-terra', 'reasoning_effort': 'medium',
    }
    assert set(report) == {'schema_version', 'safety_key', 'identities', 'metrics', 'usage', 'gate_passed'}
    assert output.err == ''


def test_live_cli_requires_both_explicit_authorizations(monkeypatch, capsys):
    monkeypatch.delenv('PARAWORKS_ALLOW_PAID_TERRA_EVAL', raising=False)
    code = main([
        '--fixture', str(FIXTURE), '--mode', 'live-openai', '--aggregate-only',
        '--allow-paid-provider-call',
    ])
    assert code == 2
    assert capsys.readouterr().out == ''


def test_metric_failure_returns_three_without_echoing_fixture(tmp_path, capsys):
    data = json.loads(FIXTURE.read_text(encoding='utf-8'))
    data['cases'][2]['predicted_approval'] = True
    fixture = tmp_path / 'redacted-fixture.json'
    fixture.write_text(json.dumps(data), encoding='utf-8')
    assert main(['--fixture', str(fixture), '--mode', 'fake', '--aggregate-only']) == 3
    report = capsys.readouterr()
    assert 'proposal_not_decision' not in report.out
    assert str(fixture) not in report.out


def test_invalid_fixture_returns_two_without_path_or_contents(tmp_path, capsys):
    fixture = tmp_path / 'private-name.json'
    fixture.write_text('{', encoding='utf-8')
    assert main(['--fixture', str(fixture), '--mode', 'fake', '--aggregate-only']) == 2
    output = capsys.readouterr()
    assert output.out == ''
    assert 'private-name' not in output.err


def test_dual_authorized_live_cli_uses_exact_langchain_boundary(
    monkeypatch, capsys
):
    data = json.loads(FIXTURE.read_text(encoding='utf-8'))
    expected_batches = [
        data['cases'][index:index + 4]
        for index in range(0, len(data['cases']), 4)
    ]

    class Structured:
        cache = False
        verbose = False

        def __init__(self):
            self.batch_index = 0

        def invoke(self, messages, config=None):
            assert messages[0][0] == 'system'
            assert config == {'callbacks': []}
            cases = expected_batches[self.batch_index]
            self.batch_index += 1
            results = []
            for ordinal, case in enumerate(cases, start=1):
                item_type = case['live_input']['item_type']
                second_field = (
                    'reason' if item_type == 'history_event' else 'result_summary'
                )
                supported = case['expected_approval']
                results.append({
                    'candidate_slot_id': f'C{ordinal:02d}',
                    'claim_results': [
                        {
                            'field_key': 'title',
                            'verdict': 'supported' if supported else 'unsupported',
                            'claim_scope': 'direct_fact',
                            'entailment_score': '0.9900',
                            'evidence_slot_ids': [f'E{ordinal:02d}'],
                        },
                        {
                            'field_key': second_field,
                            'verdict': 'supported' if supported else 'unsupported',
                            'claim_scope': 'direct_fact',
                            'entailment_score': '0.9900',
                            'evidence_slot_ids': [f'E{ordinal:02d}'],
                        },
                    ],
                    'uncertainty_codes': [],
                    'conflict_codes': [],
                })
            parsed = {'results': results}
            return {
                'parsed': parsed,
                'parsing_error': None,
                'raw': AIMessage(
                    content='',
                    usage_metadata={
                        'input_tokens': 100,
                        'output_tokens': 50,
                        'total_tokens': 150,
                    },
                ),
            }

    structured = Structured()
    captured = {}

    class Model:
        cache = False
        verbose = False

        def with_structured_output(self, schema, **kwargs):
            captured['schema'] = schema
            captured['structured_kwargs'] = kwargs
            return structured

    def builder(**kwargs):
        captured['model_kwargs'] = kwargs
        return Model()

    monkeypatch.setenv('PARAWORKS_ALLOW_PAID_TERRA_EVAL', '1')
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key-not-live')
    code = main([
        '--fixture', str(FIXTURE), '--mode', 'live-openai', '--aggregate-only',
        '--allow-paid-provider-call',
    ], model_builder=builder)
    report = json.loads(capsys.readouterr().out)

    assert code == 0
    assert report['gate_passed'] is True
    assert report['usage'] == {
        'input_tokens': 500,
        'output_tokens': 250,
        'estimated_cost_usd': '0.004000',
    }
    assert captured['model_kwargs']['model'] == 'gpt-5.6-terra'
    assert captured['model_kwargs']['reasoning_effort'] == 'medium'
    assert captured['model_kwargs']['max_retries'] == 0
    assert captured['model_kwargs']['max_completion_tokens'] == 3072
    assert captured['structured_kwargs'] == {
        'method': 'json_schema', 'strict': True, 'include_raw': True,
    }
