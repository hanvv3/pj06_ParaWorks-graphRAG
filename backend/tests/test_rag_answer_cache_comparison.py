"""Repeatable C3 measurements: real PG composition, fake generation/traversal.

Run with pytest -s for raw JSON. Nothing here measures provider price or quality.
"""

# ruff: noqa: F811 -- imported pytest fixtures
import json
import math
from contextlib import contextmanager
from decimal import Decimal
from time import perf_counter

from sqlalchemy import delete, event, select
from sqlalchemy.engine import Engine

from backend.app.agent_runtime import rag_v2_composition as composition
from backend.app.agent_runtime.rag_graph import build_company_memory_rag_answer_v2_graph
from backend.app.agent_runtime.rag_v2_state import RagRuntimeContext
from backend.app.models import AgentRun, AgentRunCostComponent, AuditLog
from backend.app.models.answer_cache import RagAnswerCacheEntry
from backend.app.rag.keyword_retriever import KeywordEvidenceRetriever
from backend.tests.test_rag_answer_cache import pg_engine  # noqa: F401
from backend.tests.test_rag_answer_cache_integration import (
    real_pg_composition,  # noqa: F401
)
from backend.tests.test_rag_v2_graph import prepare_direct_request_text
from backend.tests.test_rag_v2_provider_free_golden import _observe_external_calls


@contextmanager
def measured_sql():
    """Class hooks include new/dedicated engines, not only application sessions.

    Counts cursor executions (including SET), not driver BEGIN/COMMIT, connection
    handshakes or row fetching. Wall time includes those and service teardown.
    """
    totals = {'db_queries': 0, 'db_ms': 0.0}

    def before(conn, cursor, statement, parameters, context, executemany):
        totals['db_queries'] += 1
        context._c3_started = perf_counter()

    def after(conn, cursor, statement, parameters, context, executemany):
        totals['db_ms'] += (perf_counter() - context._c3_started) * 1000

    event.listen(Engine, 'before_cursor_execute', before)
    event.listen(Engine, 'after_cursor_execute', after)
    try:
        yield totals
    finally:
        event.remove(Engine, 'before_cursor_execute', before)
        event.remove(Engine, 'after_cursor_execute', after)


def measured_request(fixture, *, label, cache_enabled, graph_enabled=True):
    settings = fixture.settings.model_copy(
        update={
            'rag_answer_cache_enabled': cache_enabled,
            'rag_graph_enrichment_enabled': graph_enabled,
        }
    )
    sends_before = len(fixture.client.seen)
    with measured_sql() as sql:
        started = perf_counter()
        with (
            fixture.runtime.session_factory() as db,
            composition._postgres_request_services(
                db=db,
                settings=settings,
                session_factory=fixture.runtime.session_factory,
            ) as services,
        ):
            result = build_company_memory_rag_answer_v2_graph().invoke(
                {
                    'prepared_text': prepare_direct_request_text(
                        'Canonical approved knowledge',
                        key=settings.agent_runtime_fingerprint_secret.encode(),
                    )
                },
                context=RagRuntimeContext(
                    actor=fixture.actor,
                    surface='ask',
                    settings=settings,
                    services=services,
                ),
            )
        elapsed_ms = (perf_counter() - started) * 1000
    assert result['outcome'] == 'supported'
    run_id = result['run_id']
    # Verification reads are deliberately outside the request measurement.
    with fixture.runtime.session_factory() as db:
        parent = db.get(AgentRun, run_id)
        assert parent.run_record_phase == 'final'
        assert parent.total_charged_cost_usd == result['charged_cost_usd']
        components = list(
            db.scalars(
                select(AgentRunCostComponent)
                .where(AgentRunCostComponent.agent_run_id == run_id)
                .order_by(AgentRunCostComponent.component_ordinal)
            )
        )
        audits = list(
            db.scalars(
                select(AuditLog)
                .where(
                    AuditLog.target_id == str(run_id),
                    AuditLog.target_type == 'agent_run',
                    AuditLog.action.in_(
                        ('rag_answer_cache_hit', 'rag_answer_cache_finalized')
                    ),
                )
                .order_by(AuditLog.id)
            )
        )
        hit = parent.metadata_.get('answer_finalization_mode') == 'answer-cache-hit:v1'
        if hit:
            assert any(a.action == 'rag_answer_cache_hit' for a in audits)
            assert any(a.action == 'rag_answer_cache_finalized' for a in audits)
        row = {
            'label': label,
            'run_id': run_id,
            'cache_audit_ids': [a.id for a in audits],
            'cache_audit_actions': [a.action for a in audits],
            'citation_hmacs': result['evidence_projection'].citation_projection_hmacs,
            'backend': result['effective_backend'],
            'cache_hit': hit,
            'wall_ms': round(elapsed_ms, 3),
            'db_queries': sql['db_queries'],
            'db_ms': round(sql['db_ms'], 3),
            'generation_calls': len(fixture.client.seen) - sends_before,
            'simulated_cost_usd': str(parent.total_charged_cost_usd),
            'components': [
                {
                    'name': c.component,
                    'dispatches': c.dispatch_count,
                    'input_tokens': c.actual_input_tokens,
                    'output_tokens': c.actual_output_tokens,
                    'simulated_cost_usd': str(c.charged_cost_usd),
                    'charge_basis': c.charge_basis,
                }
                for c in components
            ],
        }
        generation = next(c for c in components if c.component == 'answer_generation')
        assert generation.dispatch_count == row['generation_calls'] == int(not hit)
        assert (generation.charged_cost_usd == 0) == hit
        assert result['sanitized_trace'].provider_attempt_counts['query_embedding'] == 0
        assert all(
            c.dispatch_count == 0
            for c in components
            if c.component == 'query_embedding'
        )
    return row


def summary(rows):
    def quantile(values, p):
        # Nearest rank; small samples are observations, never a performance SLO.
        return sorted(values)[max(0, math.ceil(len(values) * p) - 1)]

    return {
        'n': len(rows),
        'hit_rate': sum(r['cache_hit'] for r in rows) / len(rows),
        'generation_calls': sum(r['generation_calls'] for r in rows),
        'simulated_cost_usd': str(
            sum((Decimal(r['simulated_cost_usd']) for r in rows), Decimal(0))
        ),
        **{
            f'{metric}_{percentile}': quantile([r[metric] for r in rows], p)
            for metric in ('wall_ms', 'db_queries', 'db_ms')
            for percentile, p in (('p50', 0.5), ('p95', 0.95))
        },
    }


def test_measured_same_backend_cold_warm_and_flag_rollback(
    real_pg_composition,
    monkeypatch,
):
    fixture = real_pg_composition
    retrievals = []
    original = KeywordEvidenceRetriever.invoke

    def observe(self, request, config=None):
        retrievals.append((request.candidate_scan_limit, request.visible_limit))
        return original(self, request, config)

    monkeypatch.setattr(KeywordEvidenceRetriever, 'invoke', observe)
    rows = []

    def run(label, *, cache_enabled, graph_enabled=True):
        before = len(retrievals)
        row = measured_request(
            fixture,
            label=label,
            cache_enabled=cache_enabled,
            graph_enabled=graph_enabled,
        )
        row['fresh_retrievals'] = len(retrievals) - before
        assert row['fresh_retrievals'] >= 1
        rows.append(row)
        return row

    with _observe_external_calls() as external:
        for _iteration in range(3):
            # Reset only expendable cache entries outside measured request time.
            # Corpus, principal, budgets, model, prompt and question stay fixed.
            with fixture.runtime.engine.begin() as conn:
                conn.execute(delete(RagAnswerCacheEntry))
            baseline = run('baseline', cache_enabled=False)
            baseline_repeat = run('baseline', cache_enabled=False)
            cold = run('cold', cache_enabled=True)
            warm = run('warm', cache_enabled=True)
            assert baseline['backend'] == cold['backend'] == warm['backend'] == 'neo4j'
            assert [r['generation_calls'] for r in (baseline, cold, warm)] == [1, 1, 0]
            assert (
                not baseline['cache_hit']
                and not cold['cache_hit']
                and warm['cache_hit']
            )
            assert baseline['simulated_cost_usd'] == cold['simulated_cost_usd']
            assert baseline_repeat['generation_calls'] == 1
            assert baseline_repeat['backend'] == baseline['backend']
            assert (
                baseline['citation_hmacs']
                == baseline_repeat['citation_hmacs']
                == cold['citation_hmacs']
                == warm['citation_hmacs']
            )

        with fixture.runtime.session_factory() as db:
            old_audits = {
                a.id: (a.action, a.target_id, dict(a.metadata_))
                for a in db.scalars(select(AuditLog))
            }
            graph_keys = set(db.scalars(select(RagAnswerCacheEntry.key_hmac)))
            assert len(graph_keys) == 1
        rollback = run('cache_off_rollback', cache_enabled=False)
        assert rollback['backend'] == 'neo4j' and rollback['generation_calls'] == 1
        graph_off = run('graph_off_cold', cache_enabled=True, graph_enabled=False)
        assert graph_off['backend'] == 'deterministic_lexical'
        assert graph_off['generation_calls'] == 1 and not graph_off['cache_hit']
        graph_off_warm = run('graph_off_warm', cache_enabled=True, graph_enabled=False)
        assert graph_off_warm['backend'] == 'deterministic_lexical'
        assert graph_off_warm['generation_calls'] == 0 and graph_off_warm['cache_hit']
        restored = run('graph_restored_warm', cache_enabled=True)
        assert restored['backend'] == 'neo4j' and restored['cache_hit']

        with fixture.runtime.session_factory() as db:
            assert len(list(db.scalars(select(AgentRun)))) == len(rows)
            entries = list(db.scalars(select(RagAnswerCacheEntry)))
            assert len(entries) == 2
            assert graph_keys < {e.key_hmac for e in entries}
            for audit_id, saved in old_audits.items():
                audit = db.get(AuditLog, audit_id)
                assert (audit.action, audit.target_id, dict(audit.metadata_)) == saved

    assert external == {'external_provider_calls': 0, 'network_calls': 0}
    assert len({r['run_id'] for r in rows}) == len(rows)
    audit_ids = [aid for row in rows for aid in row['cache_audit_ids']]
    assert len(set(audit_ids)) == len(audit_ids)
    assert len(set(retrievals)) == 1  # Fixed retrieval budgets throughout.
    print(
        'C3_MEASUREMENTS='
        + json.dumps(
            {
                'cost_kind': 'simulated fake usage priced by runtime policy; no provider charges',
                'quantiles': 'nearest rank',
                'rows': rows,
                'external': external,
                'summary': {
                    label: summary([r for r in rows if r['label'] == label])
                    for label in ('baseline', 'cold', 'warm')
                },
                'cached_pair_summary': summary(
                    [r for r in rows if r['label'] in {'cold', 'warm'}]
                ),
                'query_embedding': 'keyword seed; zero dispatches, no embedding savings claim',
            },
            sort_keys=True,
        )
    )
