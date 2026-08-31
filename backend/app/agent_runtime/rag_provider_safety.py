from __future__ import annotations

import hmac
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import Connection, insert, select, update

from backend.app.agent_runtime.durable_file_authority import DurableFileAuthority
from backend.app.agent_runtime.rag_runtime_contracts import (
    AuthorizedProviderPolicySnapshot,
    RagProviderSafetyBinding,
)
from backend.app.agent_runtime.rag_safety_identity import (
    provider_safety_snapshot_identity,
    rag_identity_hmac,
)
from backend.app.models.rag_runtime import (
    RagProviderReadiness,
    RagProviderSafetyAuthority,
    RagProviderSafetyTransition,
)
from backend.app.rag.retrieval import RagPaidComponent


class RagProviderSafetyError(RuntimeError):
    pass


class RagProviderSafetyService:
    """Durable exact-two provider-family safety latch.

    The external envelope is written first for blocking transitions so a DB
    rollback can never make another worker dispatch through a known hazard.
    """

    def __init__(
        self,
        *,
        latch_path: str | Path,
        identity_secret: bytes,
        designated_environment_id: str,
    ) -> None:
        if type(identity_secret) is not bytes or not identity_secret:
            raise ValueError('provider safety signer is required')
        if type(designated_environment_id) is not str or not designated_environment_id.strip():
            raise ValueError('provider safety environment is required')
        self._authority = DurableFileAuthority(latch_path)
        self._secret = identity_secret
        self._environment = designated_environment_id

    @staticmethod
    def validate_disabled_path(path: str | Path) -> None:
        DurableFileAuthority.validate_configured_path(path)

    def _digest(self, envelope: dict[str, object]) -> str:
        unsigned = dict(envelope)
        unsigned.pop('envelope_digest', None)
        return rag_identity_hmac(
            unsigned,
            secret=self._secret,
            schema_version='rag-provider-safety-envelope:v1',
            policy_version='rag-provider-safety:v1',
        )

    @staticmethod
    def _family(snapshot: AuthorizedProviderPolicySnapshot) -> dict[str, object]:
        return {
            'component': snapshot.component,
            'provider': snapshot.provider,
            'model': snapshot.model,
            'reasoning_or_config_identity': snapshot.reasoning_or_config_identity,
            'authorized_model_config_version': snapshot.authorized_model_config_version,
            'authorized_model_config_snapshot_hmac': snapshot.authorized_model_config_snapshot_hmac,
            'authorized_cost_policy_version': snapshot.authorized_cost_policy_version,
            'authorized_token_estimator_version': snapshot.authorized_token_estimator_version,
            'fingerprint_key_version': snapshot.fingerprint_key_version,
            'fingerprint_key_material_verifier': snapshot.fingerprint_key_material_verifier,
            'authorized_policy_snapshot_hmac': snapshot.authorized_policy_snapshot_hmac,
            'state': 'ready',
            'state_version': 1,
            'family_safety_generation': 0,
        }

    def bootstrap(
        self,
        connection: Connection,
        snapshots: tuple[
            AuthorizedProviderPolicySnapshot, AuthorizedProviderPolicySnapshot
        ],
    ) -> None:
        if (
            type(snapshots) is not tuple
            or len(snapshots) != 2
            or tuple(item.component for item in snapshots)
            != ('query_embedding', 'answer_generation')
        ):
            raise RagProviderSafetyError('exactly two ordered provider families are required')
        envelope: dict[str, object] = {
            'authority_uuid': str(uuid4()),
            'designated_environment_id': self._environment,
            'global_safety_generation': 0,
            'families': [self._family(item) for item in snapshots],
        }
        envelope['envelope_digest'] = self._digest(envelope)
        self._authority.write(envelope)
        try:
            authority_uuid = envelope['authority_uuid']
            digest = envelope['envelope_digest']
            connection.execute(insert(RagProviderSafetyAuthority).values(
                id=1,
                authority_uuid=authority_uuid,
                designated_environment_id=self._environment,
                global_safety_generation=0,
                envelope_digest=digest,
                fingerprint_key_version=snapshots[0].fingerprint_key_version,
                fingerprint_key_material_verifier=snapshots[0].fingerprint_key_material_verifier,
            ))
            for snapshot in snapshots:
                connection.execute(insert(RagProviderReadiness).values(
                    authority_id=1,
                    component=snapshot.component,
                    provider=snapshot.provider,
                    model=snapshot.model,
                    reasoning_or_config_identity=snapshot.reasoning_or_config_identity,
                    active=True,
                    authorized_model_config_version=snapshot.authorized_model_config_version,
                    authorized_model_config_snapshot_hmac=snapshot.authorized_model_config_snapshot_hmac,
                    authorized_cost_policy_version=snapshot.authorized_cost_policy_version,
                    authorized_token_estimator_version=snapshot.authorized_token_estimator_version,
                    authorized_fingerprint_key_version=snapshot.fingerprint_key_version,
                    authorized_fingerprint_key_material_verifier=snapshot.fingerprint_key_material_verifier,
                    authorized_policy_snapshot_hmac=snapshot.authorized_policy_snapshot_hmac,
                    state='ready',
                    state_version=1,
                    family_safety_generation=0,
                ))
            connection.execute(insert(RagProviderSafetyTransition).values(
                authority_id=1,
                readiness_id=None,
                global_safety_generation=0,
                transition_kind='bootstrap',
                prior_state=None,
                new_state=None,
                prior_state_version=None,
                new_state_version=None,
                prior_family_safety_generation=None,
                new_family_safety_generation=None,
                envelope_digest=digest,
                reviewed_transition_reference_hmac=digest,
                actor_subject_hmac=None,
                agent_run_id=None,
            ))
            connection.commit()
        except Exception:
            connection.rollback()
            raise RagProviderSafetyError('provider safety database bootstrap failed') from None

    def _read(self) -> dict[str, object]:
        envelope = self._authority.read()
        digest = envelope.get('envelope_digest')
        if (
            type(digest) is not str
            or not hmac.compare_digest(digest, self._digest(envelope))
            or envelope.get('designated_environment_id') != self._environment
        ):
            raise RagProviderSafetyError('provider safety authority is inconsistent')
        families = envelope.get('families')
        if (
            type(families) is not list
            or len(families) != 2
            or [item.get('component') for item in families if type(item) is dict]
            != ['query_embedding', 'answer_generation']
        ):
            raise RagProviderSafetyError('provider safety families are inconsistent')
        return envelope

    def require_ready(
        self,
        connection: Connection,
        component: RagPaidComponent,
        policy_snapshot: AuthorizedProviderPolicySnapshot,
    ) -> RagProviderSafetyBinding:
        envelope = self._read()
        family = envelope['families'][0 if component == 'query_embedding' else 1]  # type: ignore[index]
        expected = self._family(policy_snapshot)
        for key in (
            'component', 'provider', 'model', 'reasoning_or_config_identity',
            'authorized_model_config_version', 'authorized_model_config_snapshot_hmac',
            'authorized_cost_policy_version', 'authorized_token_estimator_version',
            'fingerprint_key_version', 'fingerprint_key_material_verifier',
            'authorized_policy_snapshot_hmac',
        ):
            if family.get(key) != expected[key]:
                raise RagProviderSafetyError('provider family rebind is required')
        if family.get('state') != 'ready':
            raise RagProviderSafetyError('provider family is blocked')
        authority_row = connection.execute(
            select(RagProviderSafetyAuthority.__table__).where(
                RagProviderSafetyAuthority.id == 1
            )
        ).mappings().one_or_none()
        readiness_rows = connection.execute(
            select(RagProviderReadiness.__table__)
            .where(RagProviderReadiness.active.is_(True))
            .order_by(RagProviderReadiness.component)
        ).mappings().all()
        by_component = {row['component']: row for row in readiness_rows}
        db_family = by_component.get(component)
        if (
            authority_row is None
            or len(readiness_rows) != 2
            or set(by_component) != {'query_embedding', 'answer_generation'}
            or authority_row['authority_uuid'] != envelope['authority_uuid']
            or authority_row['designated_environment_id'] != self._environment
            or authority_row['global_safety_generation']
            != envelope['global_safety_generation']
            or authority_row['envelope_digest'] != envelope['envelope_digest']
            or db_family is None
            or any(db_family[key] != family[key] for key in (
                'component', 'provider', 'model', 'reasoning_or_config_identity',
                'authorized_model_config_version',
                'authorized_model_config_snapshot_hmac',
                'authorized_cost_policy_version',
                'authorized_token_estimator_version',
                'authorized_policy_snapshot_hmac', 'state', 'state_version',
                'family_safety_generation',
            ))
        ):
            raise RagProviderSafetyError('provider safety DB/external authority drift')
        digest = envelope['envelope_digest']
        snapshot_hmac = provider_safety_snapshot_identity(
            {
                'authority_uuid': envelope['authority_uuid'],
                'global_safety_generation': envelope['global_safety_generation'],
                'family': family,
                'envelope_digest': digest,
            }, secret=self._secret,
        )
        return RagProviderSafetyBinding(
            policy_snapshot=policy_snapshot,
            authority_uuid=UUID(envelope['authority_uuid']),  # type: ignore[arg-type]
            designated_environment_id=self._environment,
            envelope_digest=digest,  # type: ignore[arg-type]
            fingerprint_key_version=policy_snapshot.fingerprint_key_version,
            fingerprint_key_material_verifier=policy_snapshot.fingerprint_key_material_verifier,
            global_safety_generation=envelope['global_safety_generation'],  # type: ignore[arg-type]
            family_state='ready',
            family_state_version=family['state_version'],
            family_safety_generation=family['family_safety_generation'],
            provider_safety_snapshot_hmac=snapshot_hmac,
        )

    revalidate_before_send = require_ready
    revalidate_after_response = require_ready

    def _block(
        self,
        connection: Connection,
        component: RagPaidComponent,
        state: str,
        *,
        agent_run_id: int,
        input_tokens: int,
        output_tokens: int,
        cost_usd: Decimal,
    ) -> None:
        def transition(envelope: dict[str, object]) -> dict[str, object]:
            digest = envelope.get('envelope_digest')
            if type(digest) is not str or not hmac.compare_digest(
                digest, self._digest(envelope)
            ):
                raise RagProviderSafetyError('provider safety authority is inconsistent')
            families = envelope['families']
            family = families[0 if component == 'query_embedding' else 1]
            family['state'] = state
            family['state_version'] += 1
            family['family_safety_generation'] += 1
            family['overrun_agent_run_id'] = agent_run_id
            envelope['global_safety_generation'] += 1
            envelope['envelope_digest'] = self._digest(envelope)
            return envelope

        envelope = self._authority.update(transition)
        try:
            row = connection.execute(
                select(RagProviderReadiness.__table__).where(
                    RagProviderReadiness.component == component,
                    RagProviderReadiness.active.is_(True),
                )
            ).mappings().one()
            now = datetime.now(UTC)
            connection.execute(update(RagProviderSafetyAuthority).where(
                RagProviderSafetyAuthority.id == 1
            ).values(
                global_safety_generation=envelope['global_safety_generation'],
                envelope_digest=envelope['envelope_digest'],
                updated_at=now,
            ))
            connection.execute(update(RagProviderReadiness).where(
                RagProviderReadiness.id == row['id']
            ).values(
                state=state,
                state_version=row['state_version'] + 1,
                family_safety_generation=row['family_safety_generation'] + 1,
                overrun_agent_run_id=agent_run_id,
                overrun_input_tokens=input_tokens,
                overrun_output_tokens=output_tokens,
                overrun_cost_usd=cost_usd,
                overrun_observed_at=now,
                updated_at=now,
            ))
            connection.execute(insert(RagProviderSafetyTransition).values(
                authority_id=1,
                readiness_id=row['id'],
                global_safety_generation=envelope['global_safety_generation'],
                transition_kind=(
                    'block_overrun' if state == 'blocked_overrun'
                    else 'block_remediation'
                ),
                prior_state=row['state'],
                new_state=state,
                prior_state_version=row['state_version'],
                new_state_version=row['state_version'] + 1,
                prior_family_safety_generation=row['family_safety_generation'],
                new_family_safety_generation=row['family_safety_generation'] + 1,
                envelope_digest=envelope['envelope_digest'],
                reviewed_transition_reference_hmac=envelope['envelope_digest'],
                actor_subject_hmac=None,
                agent_run_id=agent_run_id,
            ))
            connection.commit()
        except Exception:
            connection.rollback()
            raise RagProviderSafetyError(
                'external provider blocker persisted but DB transition failed'
            ) from None

    def block_overrun(
        self,
        connection: Connection,
        component: RagPaidComponent,
        *,
        agent_run_id: int,
        input_tokens: int,
        output_tokens: int,
        cost_usd: Decimal,
    ) -> None:
        self._block(
            connection, component, 'blocked_overrun', agent_run_id=agent_run_id,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=cost_usd,
        )

    def block_remediation(
        self,
        connection: Connection,
        component: RagPaidComponent,
        *,
        agent_run_id: int,
        input_tokens: int,
        output_tokens: int,
        cost_usd: Decimal,
    ) -> None:
        self._block(
            connection, component, 'blocked_remediation', agent_run_id=agent_run_id,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
