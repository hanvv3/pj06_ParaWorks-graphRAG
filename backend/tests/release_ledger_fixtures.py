"""Deterministic row fixtures; never establish production provider authority."""

from contextlib import nullcontext
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import patch

from sqlalchemy import insert, select, update

from backend.app.models.rag_runtime import (
    RagProviderReadiness,
    RagProviderSafetyAuthority,
)
from backend.app.rag.release_ledger import (
    ReleaseRowPrimaryKey,
    release_observation_projection_hmac,
    release_row_identity_hmac,
    release_runtime_mutation_hmac,
)


def provider_rows():
    now = datetime(2026, 9, 13, tzinfo=UTC)
    authority = {
        'id': 1,
        'authority_uuid': '33333333-3333-3333-3333-333333333333',
        'designated_environment_id': 'test-only',
        'global_safety_generation': 0,
        'envelope_digest': '8' * 64,
        'fingerprint_key_version': 'v1',
        'fingerprint_key_material_verifier': '1' * 64,
        'created_at': now,
        'updated_at': now,
    }
    result = [
        (
            ReleaseRowPrimaryKey(
                'provider_safety_authority',
                {'authority_uuid': authority['authority_uuid']},
            ),
            authority,
        )
    ]
    for index, component in enumerate(('query_embedding', 'answer_generation')):
        query = component == 'query_embedding'
        row = {
            'id': index + 1,
            'authority_id': 1,
            'component': component,
            'provider': 'openai',
            'model': 'text-embedding-3-small' if query else 'gpt-5.4-mini-2026-03-17',
            'reasoning_or_config_identity': 'dimensions:1536' if query else 'low',
            'active': True,
            'authorized_model_config_version': 'rag-query-embedding-config:v1'
            if query
            else 'rag-answer-model-config:v1',
            'authorized_model_config_snapshot_hmac': '1' * 64,
            'authorized_cost_policy_version': 'rag-query-embedding-cost:v1'
            if query
            else 'rag-answer-cost:v1',
            'authorized_token_estimator_version': 'openai-cl100k-text-embedding-3-small:v1'
            if query
            else 'openai-o200k-rag-answer:v1',
            'authorized_fingerprint_key_version': 'v1',
            'authorized_fingerprint_key_material_verifier': '1' * 64,
            'authorized_policy_snapshot_hmac': '2' * 64,
            'state': 'ready',
            'state_version': 1,
            'family_safety_generation': 0,
            'overrun_agent_run_id': None,
            'overrun_input_tokens': None,
            'overrun_output_tokens': None,
            'overrun_cost_usd': None,
            'overrun_observed_at': None,
            'reset_by': None,
            'reset_at': None,
            'reviewed_gate_reference_hmac': '3' * 64,
            'created_at': now,
            'updated_at': now,
        }
        key = {
            'component': component,
            'provider_bytes': row['provider'],
            'model_bytes': row['model'],
            'reasoning_or_config_identity_bytes': row['reasoning_or_config_identity'],
        }
        result.append((ReleaseRowPrimaryKey('provider_readiness', key), row))
    return result


def observation(row, snapshot, secret):
    return {
        'row_kind': row.row_kind,
        'row_identity_hmac': release_row_identity_hmac(
            row.row_kind, row.primary_key, identity_secret=secret
        ),
        'row_projection_hmac': release_observation_projection_hmac(
            row.row_kind, snapshot, identity_secret=secret
        ),
    }


def provider_observations(secret):
    return sorted(
        [observation(row, snapshot, secret) for row, snapshot in provider_rows()],
        key=lambda item: (item['row_kind'], item['row_identity_hmac']),
    )


def observe_provider_fixture(connection, mutations):
    for model in (RagProviderSafetyAuthority, RagProviderReadiness):
        model.__table__.create(connection, checkfirst=True)
    for row, snapshot in provider_rows():
        table = (
            RagProviderSafetyAuthority.__table__
            if row.row_kind == 'provider_safety_authority'
            else RagProviderReadiness.__table__
        )
        if (
            connection.execute(
                select(table.c.id).where(table.c.id == snapshot['id'])
            ).first()
            is None
        ):
            connection.execute(insert(table).values(**snapshot))
        mutations.observe(row)


def complete_payload_roster(payload, secret):
    """Supply synthetic typed roster entries for schema-only payload tests."""
    identities = {
        (item['row_kind'], item['row_identity_hmac'])
        for item in payload['affected_rows'] + payload['observation_set']
    }

    def add(kind, key, projection='a' * 64):
        identity = release_row_identity_hmac(kind, key, identity_secret=secret)
        if (kind, identity) not in identities:
            payload['observation_set'].append(
                {
                    'row_kind': kind,
                    'row_identity_hmac': identity,
                    'row_projection_hmac': projection,
                }
            )
            identities.add((kind, identity))

    for row, snapshot in provider_rows():
        add(
            row.row_kind,
            row.primary_key,
            observation(row, snapshot, secret)['row_projection_hmac'],
        )
    common = {
        key: payload[key] for key in ('ledger_uuid', 'ledger_epoch', 'approval_id_hmac')
    }
    for index in range(payload['case_claim_count']):
        case_id = (
            payload['case_id_hmac']
            if index == 0 and payload['case_id_hmac']
            else f'{index + 100:064x}'
        )
        add('case', {**common, 'case_id_hmac': case_id})
        add('agent_run', {'agent_run_id': 41 + index})
        for component in ('query_embedding', 'answer_generation'):
            add('cost_component', {'agent_run_id': 41 + index, 'component': component})
            count = (
                payload['embedding_dispatch_count']
                if component == 'query_embedding'
                else payload['generation_dispatch_count']
            )
            if index < count:
                add(
                    'dispatch',
                    {**common, 'case_id_hmac': case_id, 'component': component},
                )
    payload['observation_set'].sort(
        key=lambda item: (item['row_kind'], item['row_identity_hmac'])
    )
    for item in payload['affected_rows']:
        if item['row_kind'] in {'agent_run', 'cost_component'}:
            item.setdefault('row_mutation_hmac', 'a' * 64)


def sign_runtime_plans(payload, mutations, connection, secret):
    """Sign declared literals against real before-images, never execute SQL."""
    for plan in mutations._plans:
        if plan.row.row_kind not in {'agent_run', 'cost_component'}:
            continue
        before = mutations._snapshot(connection, plan.row)
        after = {
            **(before or {}),
            **{str(key): value.value for key, value in plan.statement._values.items()},
        }
        identity = release_row_identity_hmac(
            plan.row.row_kind, plan.row.primary_key, identity_secret=secret
        )
        entry = next(
            item
            for item in payload['affected_rows']
            if item['row_kind'] == plan.row.row_kind
            and item['row_identity_hmac'] == identity
        )
        entry['row_mutation_hmac'] = (
            'a' * 64
            if before is None and plan.statement.is_update
            else release_runtime_mutation_hmac(
                plan.row.row_kind, before, after, identity_secret=secret
            )
        )


def row_key(kind, row):
    if kind == 'agent_run':
        return ReleaseRowPrimaryKey(kind, {'agent_run_id': row['id']})
    if kind == 'cost_component':
        return ReleaseRowPrimaryKey(
            kind, {key: row[key] for key in ('agent_run_id', 'component')}
        )
    if kind.startswith('provider_'):
        if kind == 'provider_safety_authority':
            return ReleaseRowPrimaryKey(kind, {'authority_uuid': row['authority_uuid']})
        return ReleaseRowPrimaryKey(
            kind,
            {
                'component': row['component'],
                'provider_bytes': row['provider'],
                'model_bytes': row['model'],
                'reasoning_or_config_identity_bytes': row[
                    'reasoning_or_config_identity'
                ],
            },
        )
    keys = ('ledger_uuid', 'ledger_epoch', 'approval_id_hmac')
    if kind in {'case', 'dispatch'}:
        keys += ('case_id_hmac',)
    if kind == 'dispatch':
        keys += ('component',)
    return ReleaseRowPrimaryKey(kind, {key: row[key] for key in keys})


def claim_manifest_fixture(*, provider_records=None, query_reserves=()):
    """Independent reviewed fake policy inputs, frozen before bootstrap/SQL plans."""
    from backend.app.agent_runtime.rag_runtime_contracts import (
        AuthorizedProviderPolicySnapshot,
    )
    from backend.app.rag.release_review import (
        FrozenCaseClaimCase,
        FrozenCaseClaimComponent,
        FrozenCaseClaimManifest,
    )

    provider_records = provider_records or [
        row for key, row in provider_rows() if key.row_kind == 'provider_readiness'
    ]
    policies = []
    for component in ('query_embedding', 'answer_generation'):
        row = next(
            row
            for row in provider_records
            if row['component'] == component and row['active']
        )
        policies.append(
            AuthorizedProviderPolicySnapshot(
                **{
                    name: row[
                        'authorized_' + name
                        if name.startswith('fingerprint_')
                        else name
                    ]
                    for name in AuthorizedProviderPolicySnapshot.__dataclass_fields__
                }
            )
        )
    return FrozenCaseClaimManifest(
        source_manifest_hmac='9' * 64,
        cases=tuple(
            FrozenCaseClaimCase(
                ordinal=ordinal,
                case_id_hmac=f'{ordinal + 1:064x}',
                surface='ask',
                configured_backend='pgvector' if query else 'keyword',
                current_text_hmac='4' * 64,
                retrieval_query_hmac='5' * 64,
                security_scope_fingerprint='6' * 64,
                components=(
                    FrozenCaseClaimComponent(policies[0], 10 if query else 0, 0, query),
                    FrozenCaseClaimComponent(policies[1], 10, 10, Decimal('0.001000')),
                ),
            )
            for ordinal in range(30)
            for query in [
                Decimal(query_reserves[ordinal])
                if ordinal < len(query_reserves)
                else Decimal('0.000000')
            ]
        ),
    )


class ReleaseHarness:
    """Real SQL mutation/append harness; only authority transport is a test seam."""

    def __init__(
        self, engine, authority, secret, database_identity, *, query_reserves=()
    ):
        from backend.app.models.agent_runs import AgentRun
        from backend.app.models.rag_runtime import AgentRunCostComponent
        from backend.app.rag.release_ledger import RagReleaseLedger
        from backend.tests.test_rag_release_ledger import (
            _bootstrap_payload,
            _capture_bootstrap_authorization,
        )

        self.engine, self.authority, self.secret, self.database_identity = (
            engine,
            authority,
            secret,
            database_identity,
        )
        self.ledger = RagReleaseLedger(authority=authority, identity_secret=secret)
        with engine.connect() as connection:
            AgentRun.__table__.create(connection, checkfirst=True)
            AgentRunCostComponent.__table__.create(connection, checkfirst=True)
            self.snapshot = authority.initialize(
                connection,
                database_identity=database_identity,
                review_envelope_hmac='1' * 64,
                review_nonce_hmac='2' * 64,
            )
        payload = _bootstrap_payload(self.snapshot)
        from backend.app.rag.release_ledger import _derived_payload_rows

        payload['affected_rows'] = [
            {'row_kind': kind, 'row_identity_hmac': identity}
            for kind, identity in _derived_payload_rows(payload, identity_secret=secret)
        ]
        with engine.connect() as connection:
            mutations = _capture_bootstrap_authorization(
                connection, self.ledger, payload
            )
            # Read the exact provider rows (including an in-process real Task22 peer).
            mutations._observation_rows.clear()
            payload['observation_set'] = []
            for kind in ('provider_safety_authority', 'provider_readiness'):
                table, _aliases = mutations._table(kind)
                for record in connection.execute(select(table)).mappings():
                    row = dict(record)
                    key = row_key(kind, row)
                    mutations.observe(key)
                    payload['observation_set'].append(observation(key, row, secret))
                    if kind == 'provider_safety_authority':
                        payload['provider_safety_envelope_digest'] = row[
                            'envelope_digest'
                        ]
            payload['observation_set'].sort(
                key=lambda item: (item['row_kind'], item['row_identity_hmac'])
            )
            mutations._plans[0] = type(mutations._plans[0])(
                mutations._plans[0].statement.values(
                    provider_safety_envelope_digest=payload[
                        'provider_safety_envelope_digest'
                    ]
                ),
                mutations._plans[0].row,
            )
            self.manifest = claim_manifest_fixture(
                provider_records=[
                    dict(row)
                    for row in connection.execute(
                        select(RagProviderReadiness.__table__)
                    ).mappings()
                ],
                query_reserves=query_reserves,
            )
            self.approved_manifest = deepcopy(self.manifest)
            self.source_verifier = self._verify_test_approved_source
            mutations._plans[0] = type(mutations._plans[0])(
                mutations._plans[0].statement.values(
                    manifest_hmac=self.manifest.source_manifest_hmac
                ),
                mutations._plans[0].row,
            )
            self.snapshot = self.ledger.append(
                connection,
                payload,
                actual_mutations=mutations,
                database_identity=database_identity,
            )
        self.base = payload

    def _verify_test_approved_source(
        self,
        connection,
        *,
        manifest,
        source_binding,
        authorization,
        identity_secret,
        barrier_guard=None,
    ):
        """Explicit fake approved source; never installed outside a test call.

        Task23's synthetic case distributions intentionally differ from the live
        fixture. Full source issuance/authority refusal uses separate Task24
        tests; all existing literal SQL/image checks continue to run unchanged.
        """
        from backend.app.rag.release_review import _manifest_payload, _refuse

        if (
            connection.engine is not self.engine
            or identity_secret != self.secret
            or _manifest_payload(manifest) != _manifest_payload(self.approved_manifest)
            or authorization['manifest_hmac']
            != self.approved_manifest.source_manifest_hmac
        ):
            _refuse()

    def records(self, kind):
        from backend.app.rag.release_ledger import RagReleaseMutationSet

        table, _aliases = RagReleaseMutationSet._table(kind)
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(select(table)).mappings()]

    def prepare(
        self,
        connection,
        kind,
        changes,
        *,
        case=None,
        component=None,
        outcome=None,
        payload_overrides=None,
    ):
        from backend.app.rag.release_ledger import _derived_payload_rows

        payload = {
            **self.base,
            'transition_kind': kind,
            'outcome': outcome,
            'from_generation': self.snapshot.generation,
            'to_generation': self.snapshot.generation + 1,
        }
        mutations = self.ledger.mutation_set(connection)
        old_auth = self.records('authorization')[0]
        auth = old_auth
        for row_kind, before, after in changes:
            key = row_key(row_kind, after)
            table, aliases = mutations._table(row_kind)
            statement = (
                insert(table).values(**after)
                if before is None
                else update(table)
                .where(
                    *(
                        table.c[aliases.get(field, field)] == value
                        for field, value in key.primary_key.items()
                    )
                )
                .values(
                    **{
                        field: value
                        for field, value in after.items()
                        if before[field] != value
                    }
                )
            )
            mutations.plan(statement, key)
            if row_kind == 'authorization':
                auth = after
        payload.update(
            authorization_state_before=old_auth['state'],
            authorization_state_after=auth['state'],
        )
        for field in (
            'case_claim_count',
            'embedding_dispatch_count',
            'generation_dispatch_count',
            'total_dispatch_count',
            'execution_process_instance_hmac',
            'execution_runner_fence_hmac',
        ):
            payload[field] = auth[field]
        payload['authorization_reserved_cost_usd'] = format(
            Decimal(str(auth['reserved_cost_usd'])), '.6f'
        )
        payload['authorization_charged_cost_usd'] = format(
            Decimal(str(auth['charged_cost_usd'])), '.6f'
        )
        if case is not None:
            previous = next(
                (
                    before
                    for row_kind, before, after in changes
                    if row_kind == 'case'
                    and after['case_id_hmac'] == case['case_id_hmac']
                ),
                case,
            )
            payload.update(
                case_id_hmac=case['case_id_hmac'],
                case_state_before=None if previous is None else previous['state'],
                case_state_after=case['state'],
                case_projection_hmac=case['case_projection_hmac'],
                runtime_agent_run_id_hmac=case['runtime_agent_run_id_hmac'],
            )
            for part in ('embedding', 'generation', 'total'):
                payload[f'case_{part}_reserved_cost_usd'] = format(
                    Decimal(str(case[f'{part}_reserved_cost_usd'])), '.6f'
                )
        if component is not None:
            prior, dispatch = next(
                (before, after)
                for row_kind, before, after in changes
                if row_kind == 'dispatch'
            )
            payload.update(
                component=component,
                dispatch_state_before='not_attempted'
                if prior is None
                else prior['state'],
                dispatch_state_after=dispatch['state'],
                dispatch_count_before=0 if prior is None else prior['dispatch_count'],
                dispatch_count_after=dispatch['dispatch_count'],
                dispatch_fence_hmac=dispatch['dispatch_fence_hmac'],
                reserved_cost_usd=format(dispatch['reserved_cost_usd'], '.6f'),
                charged_cost_usd=format(dispatch['charged_cost_usd'], '.6f'),
                charge_basis_after=dispatch['charge_basis'],
            )
        changed_keys = [plan.row for plan in mutations._plans]
        payload['observation_set'] = []
        for row_kind in (
            'authorization',
            'case',
            'dispatch',
            'agent_run',
            'cost_component',
            'provider_safety_authority',
            'provider_readiness',
            'quality_report',
        ):
            table, _aliases = mutations._table(row_kind)
            for record in connection.execute(select(table)).mappings():
                row = dict(record)
                key = row_key(row_kind, row)
                if key not in changed_keys:
                    mutations.observe(key)
                    payload['observation_set'].append(
                        observation(key, row, self.secret)
                    )
                if row_kind == 'provider_safety_authority':
                    payload['provider_safety_envelope_digest'] = row['envelope_digest']
        payload['observation_set'].sort(
            key=lambda item: (item['row_kind'], item['row_identity_hmac'])
        )
        internal = [
            (kind, identity)
            for kind, identity in _derived_payload_rows(
                payload, identity_secret=self.secret
            )
            if kind in {'release_ledger', 'release_transition'}
        ]
        payload['affected_rows'] = sorted(
            [
                {
                    'row_kind': key.row_kind,
                    'row_identity_hmac': release_row_identity_hmac(
                        key.row_kind, key.primary_key, identity_secret=self.secret
                    ),
                }
                for key in changed_keys
            ]
            + [
                {'row_kind': kind, 'row_identity_hmac': identity}
                for kind, identity in internal
            ],
            key=lambda item: (item['row_kind'], item['row_identity_hmac']),
        )
        payload.update(payload_overrides or {})
        sign_runtime_plans(payload, mutations, connection, self.secret)
        return payload, mutations

    def append(self, kind, changes, *, approved_case_claim=None, **kwargs):
        # Task24-B revalidates the source under append's barrier as well as
        # during preparation. Keep this existing synthetic-roster fake scoped
        # across both checks; no test seam is installed in production.
        verifier_context = (
            patch(
                'backend.app.rag.release_review.require_approved_case_source',
                self.source_verifier,
            )
            if self.source_verifier is not None
            else nullcontext()
        )
        with verifier_context, self.engine.connect() as connection:
            payload, mutations = self.prepare(connection, kind, changes, **kwargs)
            self.snapshot = self.ledger.append(
                connection,
                payload,
                actual_mutations=mutations,
                database_identity=self.database_identity,
                approved_case_claim=approved_case_claim,
            )
        return payload

    def claim(
        self, ordinal=0, *, generation_reserve='0.001000', query_reserve='0.000000'
    ):
        from backend.app.agent_runtime.rag_safety_identity import rag_identity_hmac
        from backend.app.rag.release_review import prepare_case_claim_projection

        before = self.records('authorization')[0]
        reserve = Decimal(generation_reserve)
        query_budget = Decimal(query_reserve)
        auth = {
            **before,
            'state': 'started',
            'case_claim_count': ordinal + 1,
            'execution_process_instance_hmac': 'd' * 64,
            'execution_runner_fence_hmac': 'e' * 64,
            'reserved_cost_usd': before['reserved_cost_usd'] + reserve + query_budget,
        }
        projection_payload = {
            **self.base,
            'from_generation': self.snapshot.generation,
            'case_id_hmac': f'{ordinal + 1:064x}',
            'execution_process_instance_hmac': auth['execution_process_instance_hmac'],
            'execution_runner_fence_hmac': auth['execution_runner_fence_hmac'],
        }
        verifier_context = (
            patch(
                'backend.app.rag.release_review.require_approved_case_source',
                self.source_verifier,
            )
            if self.source_verifier is not None
            else nullcontext()
        )
        with verifier_context, self.engine.connect() as connection:
            projection = prepare_case_claim_projection(
                connection,
                manifest=self.manifest,
                payload=projection_payload,
                identity_secret=self.secret,
            )
        parent, query, generation = projection.runtime_images
        case = {
            **{
                key: before[key]
                for key in ('ledger_uuid', 'ledger_epoch', 'approval_id_hmac')
            },
            'case_id_hmac': f'{ordinal + 1:064x}',
            'manifest_ordinal': ordinal,
            'state': 'claimed',
            'case_projection_hmac': None,
            'runtime_agent_run_id_hmac': rag_identity_hmac(
                {'agent_run_id': parent['id']},
                secret=self.secret,
                schema_version='rag-runtime-agent-run-id:v1',
                policy_version='rag-run:v2',
            ),
            'embedding_reserved_cost_usd': query_budget,
            'generation_reserved_cost_usd': reserve,
            'total_reserved_cost_usd': reserve + query_budget,
        }
        self.append(
            'case_claim',
            [
                ('authorization', before, auth),
                ('case', None, case),
                ('agent_run', None, parent),
                ('cost_component', None, query),
                ('cost_component', None, generation),
            ],
            case=case,
            approved_case_claim=projection,
        )

    def claim_generation(self, component='answer_generation'):
        before = self.records('authorization')[0]
        case = self.records('case')[-1]
        cost = next(
            row
            for row in self.records('cost_component')[-2:]
            if row['component'] == component
        )
        counter = (
            'generation_dispatch_count'
            if component == 'answer_generation'
            else 'embedding_dispatch_count'
        )
        auth = {
            **before,
            counter: before[counter] + 1,
            'total_dispatch_count': before['total_dispatch_count'] + 1,
            'charged_cost_usd': before['charged_cost_usd'] + cost['reserved_cost_usd'],
        }
        child = {
            **cost,
            'dispatch_state': 'dispatching',
            'attempted': True,
            'dispatch_count': 1,
            'process_instance_hmac': 'd' * 64,
            'dispatch_fence_hmac': '5' * 64,
            'charge_basis': 'reserved',
            'charged_cost_usd': cost['reserved_cost_usd'],
        }
        dispatch = {
            **{
                key: case[key]
                for key in (
                    'ledger_uuid',
                    'ledger_epoch',
                    'approval_id_hmac',
                    'case_id_hmac',
                )
            },
            'component': component,
            'state': 'dispatching',
            'dispatch_count': 1,
            'dispatch_fence_hmac': '5' * 64,
            'reserved_cost_usd': cost['reserved_cost_usd'],
            'charged_cost_usd': cost['reserved_cost_usd'],
            'charge_basis': 'reserved',
        }
        return self.append(
            'component_claim',
            [
                ('authorization', before, auth),
                ('cost_component', cost, child),
                ('dispatch', None, dispatch),
            ],
            case=case,
            component=component,
        )

    def generation_outcome(self, *, overrun=False):
        auth = self.records('authorization')[0]
        case = self.records('case')[-1]
        run = self.records('agent_run')[-1]
        cost = self.records('cost_component')[-1]
        dispatch = self.records('dispatch')[-1]
        actual = Decimal('0.000400')
        return self.append(
            'component_outcome',
            [
                (
                    'authorization',
                    auth,
                    {
                        **auth,
                        'charged_cost_usd': auth['charged_cost_usd']
                        - cost['charged_cost_usd']
                        + actual,
                    },
                ),
                (
                    'agent_run',
                    run,
                    {
                        **run,
                        'run_record_phase': 'cost_finalized_pending_projection',
                        'projection_owner_fence_hmac': 'e' * 64,
                        'total_charged_cost_usd': actual,
                    },
                ),
                (
                    'cost_component',
                    cost,
                    {
                        **cost,
                        'dispatch_state': 'terminal',
                        'actual_input_tokens': 1,
                        'actual_output_tokens': 2,
                        'charged_cost_usd': actual,
                        'charge_basis': 'actual',
                        'terminal_outcome': 'component_succeeded',
                        'overrun': overrun,
                    },
                ),
                (
                    'dispatch',
                    dispatch,
                    {
                        **dispatch,
                        'state': 'terminal',
                        'charged_cost_usd': actual,
                        'charge_basis': 'actual',
                    },
                ),
            ],
            case=case,
            component='answer_generation',
            outcome='component_succeeded',
        )

    def fail_case(self, kind='case_failure', outcome='persistence_failed'):
        case = self.records('case')[-1]
        run = self.records('agent_run')[-1]
        failed = {**case, 'state': 'failed'}
        changes = [
            ('case', case, failed),
            (
                'agent_run',
                run,
                {
                    **run,
                    'status': 'failed',
                    'run_record_phase': 'final',
                    'completed_at': datetime.now(UTC),
                },
            ),
        ]
        if kind.startswith('authorization_abort'):
            auth = self.records('authorization')[0]
            changes.append(
                ('authorization', auth, {**auth, 'state': 'aborted_provider_safety'})
            )
        for child in self.records('cost_component')[-2:]:
            if child['dispatch_state'] == 'not_attempted':
                changes.append(
                    (
                        'cost_component',
                        child,
                        {
                            **child,
                            'dispatch_state': 'terminal',
                            'reserved_input_tokens': 0,
                            'reserved_output_tokens': 0,
                            'reserved_cost_usd': Decimal(0),
                        },
                    )
                )
        return self.append(kind, changes, case=failed, outcome=outcome)
