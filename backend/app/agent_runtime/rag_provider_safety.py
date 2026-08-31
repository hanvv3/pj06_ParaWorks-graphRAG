from __future__ import annotations

import hashlib
import hmac
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import Connection, insert, select, update

from backend.app.agent_runtime.durable_file_authority import DurableFileAuthority
from backend.app.agent_runtime.fingerprints import canonical_json_bytes
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
        self._latch_path = DurableFileAuthority.validate_configured_path(latch_path)
        self._authority = DurableFileAuthority.open_runtime(self._latch_path)
        self._secret = identity_secret
        self._environment = designated_environment_id

    @staticmethod
    def validate_disabled_path(path: str | Path) -> None:
        DurableFileAuthority.validate_configured_path(path)

    def _digest(self, signed_payload: dict[str, object]) -> str:
        return rag_identity_hmac(
            signed_payload,
            secret=self._secret,
            schema_version='rag-provider-safety-latch-envelope:v1',
            policy_version='rag-provider-safety-latch-body:v1',
        )

    @staticmethod
    def _file_digest(envelope: dict[str, object]) -> str:
        return hashlib.sha256(
            b'paraworks:provider-safety-envelope-file:v1\x00'
            + canonical_json_bytes(envelope)
        ).hexdigest()

    def _wrap(
        self, body: dict[str, object], *, key_version: str, key_verifier: str
    ) -> dict[str, object]:
        signed_payload = {
            'body': body,
            'fingerprint_key_material_verifier': key_verifier,
            'fingerprint_key_version': key_version,
        }
        return {
            'signed_payload': signed_payload,
            'hmac_sha256': self._digest(signed_payload),
        }

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
        body: dict[str, object] = {
            'authority_uuid': str(uuid4()),
            'designated_environment_id': self._environment,
            'global_safety_generation': 0,
            'families': [self._family(item) for item in snapshots],
            'latch_schema_version': 'rag-provider-safety-latch-body:v1',
        }
        envelope = self._wrap(
            body,
            key_version=snapshots[0].fingerprint_key_version,
            key_verifier=snapshots[0].fingerprint_key_material_verifier,
        )
        initializer = DurableFileAuthority(self._latch_path)
        initializer.write(envelope)
        self._authority = DurableFileAuthority.open_runtime(self._latch_path)
        try:
            authority_uuid = body['authority_uuid']
            digest = self._file_digest(envelope)
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
        if set(envelope) != {'signed_payload', 'hmac_sha256'}:
            raise RagProviderSafetyError('provider safety authority is inconsistent')
        signed = envelope.get('signed_payload')
        signature = envelope.get('hmac_sha256')
        if (
            type(signed) is not dict
            or type(signature) is not str
            or not hmac.compare_digest(signature, self._digest(signed))
            or type(signed.get('body')) is not dict
        ):
            raise RagProviderSafetyError('provider safety authority is inconsistent')
        body = dict(signed['body'])
        if body.get('designated_environment_id') != self._environment:
            raise RagProviderSafetyError('provider safety authority is inconsistent')
        families = body.get('families')
        if (
            type(families) is not list
            or len(families) != 2
            or [item.get('component') for item in families if type(item) is dict]
            != ['query_embedding', 'answer_generation']
        ):
            raise RagProviderSafetyError('provider safety families are inconsistent')
        body['envelope_digest'] = self._file_digest(envelope)
        body['_fingerprint_key_version'] = signed.get('fingerprint_key_version')
        body['_fingerprint_key_material_verifier'] = signed.get(
            'fingerprint_key_material_verifier'
        )
        return body

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
            signed = envelope.get('signed_payload')
            digest = envelope.get('hmac_sha256')
            if type(digest) is not str or not hmac.compare_digest(
                digest, self._digest(signed)
            ):
                raise RagProviderSafetyError('provider safety authority is inconsistent')
            body = signed['body']
            families = body['families']
            family = families[0 if component == 'query_embedding' else 1]
            family['state'] = state
            family['state_version'] += 1
            family['family_safety_generation'] += 1
            family['overrun_agent_run_id'] = agent_run_id
            body['global_safety_generation'] += 1
            return self._wrap(
                body,
                key_version=signed['fingerprint_key_version'],
                key_verifier=signed['fingerprint_key_material_verifier'],
            )

        def commit_db(raw_envelope: dict[str, object]) -> None:
            signed = raw_envelope['signed_payload']
            envelope = signed['body']
            digest = self._file_digest(raw_envelope)
            try:
                rows = connection.execute(
                    select(RagProviderReadiness.__table__)
                    .where(RagProviderReadiness.active.is_(True))
                    .order_by(RagProviderReadiness.component)
                    .with_for_update()
                ).mappings().all()
                if len(rows) != 2:
                    raise RagProviderSafetyError('provider safety whole set is invalid')
                row = next(item for item in rows if item['component'] == component)
                now = datetime.now(UTC)
                connection.execute(update(RagProviderSafetyAuthority).where(
                    RagProviderSafetyAuthority.id == 1
                ).values(
                    global_safety_generation=envelope['global_safety_generation'],
                    envelope_digest=digest, updated_at=now,
                ))
                connection.execute(update(RagProviderReadiness).where(
                    RagProviderReadiness.id == row['id'],
                    RagProviderReadiness.state_version == row['state_version'],
                ).values(
                    state=state, state_version=row['state_version'] + 1,
                    family_safety_generation=row['family_safety_generation'] + 1,
                    overrun_agent_run_id=agent_run_id,
                    overrun_input_tokens=input_tokens,
                    overrun_output_tokens=output_tokens,
                    overrun_cost_usd=cost_usd,
                    overrun_observed_at=now, updated_at=now,
                ))
                connection.execute(insert(RagProviderSafetyTransition).values(
                    authority_id=1, readiness_id=row['id'],
                    global_safety_generation=envelope['global_safety_generation'],
                    transition_kind=(
                        'block_overrun' if state == 'blocked_overrun'
                        else 'block_remediation'
                    ), prior_state=row['state'], new_state=state,
                    prior_state_version=row['state_version'],
                    new_state_version=row['state_version'] + 1,
                    prior_family_safety_generation=row['family_safety_generation'],
                    new_family_safety_generation=row['family_safety_generation'] + 1,
                    envelope_digest=digest,
                    reviewed_transition_reference_hmac=digest,
                    actor_subject_hmac=None, agent_run_id=agent_run_id,
                ))
                connection.commit()
            except Exception:
                connection.rollback()
                raise RagProviderSafetyError(
                    'external provider blocker persisted but DB transition failed'
                ) from None

        self._authority.update(transition, after_replace=commit_db)

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
