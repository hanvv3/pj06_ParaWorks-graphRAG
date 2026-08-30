import json
from decimal import Decimal
from pathlib import Path

from langchain_core.messages import AIMessage

from backend.app.review.auto_review_extraction_compatibility import main

FIXTURE = Path(__file__).parent / 'fixtures' / 'auto_review_extraction_compat_v1.json'


def test_fake_extraction_cli_proves_exact_five_route_registry(capsys):
    assert main(['--fixture', str(FIXTURE), '--mode', 'fake', '--aggregate-only']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['safety_key'] == {
        'purpose': 'extraction', 'provider': 'openai',
        'model': 'gpt-5.4-mini-2026-03-17', 'reasoning_effort': 'none',
    }
    assert report['aggregate']['route_count'] == 5
    assert report['aggregate']['passed_route_count'] == 5
    assert Decimal(report['aggregate']['per_route_full_cap_reserve_usd']) == Decimal('0.016716')
    assert Decimal(report['aggregate']['five_route_full_cap_reserve_usd']) == Decimal('0.083580')
    assert len(report['route_identities']) == 5


def test_live_extraction_cli_requires_both_authorizations(monkeypatch, capsys):
    monkeypatch.delenv('PARAWORKS_ALLOW_PAID_EXTRACTION_EVAL', raising=False)
    assert main([
        '--fixture', str(FIXTURE), '--mode', 'live-openai', '--aggregate-only',
        '--allow-paid-provider-call',
    ]) == 2
    assert capsys.readouterr().out == ''


def test_extraction_fixture_drift_returns_metric_failure(tmp_path, capsys):
    data = json.loads(FIXTURE.read_text(encoding='utf-8'))
    data['routes'][0]['prompt_version'] = 'drifted'
    fixture = tmp_path / 'fixture.json'
    fixture.write_text(json.dumps(data), encoding='utf-8')
    assert main(['--fixture', str(fixture), '--mode', 'fake', '--aggregate-only']) == 3
    assert 'drifted' not in capsys.readouterr().out


def test_dual_authorized_live_extraction_cli_uses_all_exact_routes(
    monkeypatch, capsys
):
    captured = {'model_kwargs': [], 'schemas': [], 'calls': 0}

    class Structured:
        cache = False
        verbose = False

        def __init__(self, schema):
            self.schema = schema

        def invoke(self, messages, config=None):
            captured['calls'] += 1
            assert messages[0][0] == 'human'
            assert config == {'callbacks': []}
            if captured['calls'] % 2 == 1:
                schema_name = self.schema.__name__
                item_type = {
                    'MailDocumentExtractionResult': 'timeline_event',
                    'TimelineExtractionResult': 'timeline_event',
                    'HistoryExtractionResult': 'history_event',
                    'DecisionRecordExtractionResult': 'decision_record',
                    'TodoExtractionResult': 'todo',
                }[schema_name]
                candidate = {
                    'item_type': item_type,
                    'title': '검증용 제목',
                    'summary': '검증용 요약',
                    'confidence_score': '0.9900',
                    'uncertainty_reason': None,
                    'field_evidence_bindings': [
                        {'field_key': 'title', 'evidence_slot_id': 'S01'},
                        {'field_key': 'summary', 'evidence_slot_id': 'S02'},
                    ],
                }
                if item_type == 'timeline_event':
                    candidate['result_summary'] = '검증용 결과'
                    candidate['field_evidence_bindings'].append(
                        {'field_key': 'result_summary', 'evidence_slot_id': 'S03'}
                    )
                elif item_type == 'history_event':
                    candidate['reason'] = '검증용 이유'
                    candidate['field_evidence_bindings'].append(
                        {'field_key': 'reason', 'evidence_slot_id': 'S03'}
                    )
                elif item_type == 'decision_record':
                    candidate['decision_summary'] = '검증용 결정'
                    candidate['field_evidence_bindings'].append(
                        {'field_key': 'decision_summary', 'evidence_slot_id': 'S03'}
                    )
                else:
                    candidate['priority'] = 'high'
                    candidate['priority_reason'] = '검증용 우선순위 이유'
                    candidate['field_evidence_bindings'].extend([
                        {'field_key': 'priority', 'evidence_slot_id': 'S03'},
                        {'field_key': 'priority_reason', 'evidence_slot_id': 'S04'},
                    ])
                parsed = {
                    'result_kind': 'candidate',
                    'candidate': candidate,
                    'no_candidate_reason': None,
                }
            else:
                parsed = {
                    'result_kind': 'no_candidate',
                    'candidate': None,
                    'no_candidate_reason': 'no_relevant_evidence',
                }
            return {
                'parsed': self.schema.model_validate(parsed),
                'parsing_error': None,
                'raw': AIMessage(
                    content='',
                    usage_metadata={
                        'input_tokens': 100,
                        'output_tokens': 20,
                        'total_tokens': 120,
                    },
                ),
            }

    class Model:
        cache = False
        verbose = False

        def with_structured_output(self, schema, **kwargs):
            captured['schemas'].append((schema.__name__, kwargs))
            return Structured(schema)

    def builder(**kwargs):
        captured['model_kwargs'].append(kwargs)
        return Model()

    monkeypatch.setenv('PARAWORKS_ALLOW_PAID_EXTRACTION_EVAL', '1')
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key-not-live')
    code = main([
        '--fixture', str(FIXTURE), '--mode', 'live-openai', '--aggregate-only',
        '--allow-paid-provider-call',
    ], model_builder=builder)
    report = json.loads(capsys.readouterr().out)

    assert code == 0
    assert report['gate_passed'] is True
    assert captured['calls'] == 10
    assert len(captured['model_kwargs']) == 5
    assert {item['model'] for item in captured['model_kwargs']} == {
        'gpt-5.4-mini-2026-03-17'
    }
    assert {item['reasoning_effort'] for item in captured['model_kwargs']} == {
        'none'
    }
    assert {item['max_retries'] for item in captured['model_kwargs']} == {0}
    assert {item['max_completion_tokens'] for item in captured['model_kwargs']} == {
        2048
    }
    assert all(kwargs == {
        'method': 'json_schema', 'strict': True, 'include_raw': True,
    } for _, kwargs in captured['schemas'])
    assert report['usage'] == {
        'input_tokens': 1000,
        'output_tokens': 200,
        'estimated_cost_usd': '0.001650',
    }
