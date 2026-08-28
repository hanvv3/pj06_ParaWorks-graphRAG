from collections.abc import Mapping, Sequence

from sqlalchemy import Connection, inspect, text
from sqlalchemy.orm import Session

C5_GRAPH_VERSION = 'company-memory-review-v2.1-auto-review'

C5_TABLE_NAMES = (
    'auto_review_runtime_key_states',
    'auto_review_provider_safety_states',
    'auto_review_provider_safety_events',
    'trusted_knowledge_fingerprint_projection_states',
    'trusted_knowledge_fingerprints',
    'review_item_evidence_refs',
    'auto_review_extraction_calls',
    'auto_review_validation_calls',
    'auto_review_validations',
    'trusted_knowledge_approval_links',
    'trusted_knowledge_evidence_links',
    'assistant_message_evidence_dependencies',
    'assistant_message_knowledge_evidence_refs',
    'auto_review_rollout_states',
    'auto_review_rollout_control_events',
    'auto_review_promotion_decisions',
    'auto_review_post_audits',
    'auto_review_revocation_assessments',
    'auto_review_audit_corrections',
    'vector_serving_tombstones',
)

C5_ADDED_COLUMNS: Mapping[str, Sequence[str]] = {
    'agent_workflow_requests': (
        'fingerprint_key_material_verifier',
        'auto_review_mode',
        'auto_review_validator_provider',
        'auto_review_validator_model',
        'auto_review_reasoning_effort',
        'auto_review_validator_prompt_version',
        'auto_review_validator_output_contract_version',
        'auto_review_policy_version',
        'auto_review_cost_policy_version',
        'auto_review_extraction_cost_policy_version',
        'auto_review_token_estimator_version',
        'auto_review_extraction_token_estimator_version',
        'auto_review_tokenizer_encoding',
        'auto_review_reply_priming_tokens',
        'auto_review_framing_safety_tokens',
        'auto_review_max_input_tokens',
        'auto_review_max_output_tokens',
        'auto_review_max_candidates_per_batch',
        'auto_review_max_batches_per_workflow',
        'auto_review_max_candidates_per_workflow',
        'auto_review_max_provider_attempts',
        'auto_review_provider_timeout_seconds',
        'auto_review_provider_send_start_window_seconds',
        'auto_review_provider_attempt_lease_seconds',
        'auto_review_provider_commit_grace_seconds',
        'auto_review_validator_input_usd_per_1m',
        'auto_review_validator_output_usd_per_1m',
        'auto_review_extraction_input_usd_per_1m',
        'auto_review_extraction_output_usd_per_1m',
        'auto_review_extraction_provider',
        'auto_review_extraction_model',
        'auto_review_extraction_reasoning_effort',
        'auto_review_extraction_route_version',
        'auto_review_enforce_percentage',
        'authorized_percentage_at_launch',
        'rollout_authorization_generation',
        'validation_provider_safety_state_version',
        'extraction_provider_safety_snapshot_set_hmac',
        'rollout_control_epoch',
        'selected_extraction_agent_count',
        'extraction_plan_set_hmac',
        'extraction_max_input_chars_per_agent',
        'extraction_max_input_tokens_per_agent',
        'extraction_max_output_tokens_per_agent',
        'extraction_max_candidates_per_agent',
        'confirmed_extraction_cost_ceiling_usd',
        'confirmed_validation_cost_ceiling_usd',
        'confirmed_total_cost_ceiling_usd',
        'auto_review_budget_limit_usd',
    ),
    'agent_runs': (
        'generation_provider',
        'generation_reasoning_effort',
        'generation_route_version',
        'generation_output_contract_version',
    ),
    'review_items': (
        'agent_run_id',
        'candidate_contract_version',
        'resolution_source',
        'resolution_policy_version',
        'auto_validation_id',
        'revoked_at',
        'revoked_by_subject_hmac',
        'revoked_by_fingerprint_key_version',
        'revoked_by_fingerprint_key_material_verifier',
        'revoke_knowledge_remained_trusted',
        'revoke_document_count',
    ),
    'assistant_messages': (
        'evidence_contract_version',
        'serving_dependency_count',
    ),
    'sources': (
        'server_content_signature_schema',
        'server_content_signature',
        'connector_content_signature',
    ),
    'documents': ('current_document_version_id',),
    'document_parser_runs': (
        'server_content_signature_schema',
        'server_content_signature',
        'parser_policy_version',
        'parser_version',
        'chunk_policy_version',
    ),
    'document_chunks': ('parser_run_id',),
}


class C5SchemaCapabilityError(RuntimeError):
    pass


def c5_schema_available(bind: Connection | Session) -> bool:
    connection = _bind(bind)
    schema = inspect(connection)
    tables = set(schema.get_table_names())
    columns_by_table = {
        table_name: {column['name'] for column in schema.get_columns(table_name)}
        for table_name in C5_ADDED_COLUMNS
        if table_name in tables
    }
    present_c5_tables = set(C5_TABLE_NAMES) & tables
    present_c5_columns = {
        (table_name, column_name)
        for table_name, column_names in C5_ADDED_COLUMNS.items()
        if table_name in tables
        for column_name in column_names
        if column_name in columns_by_table[table_name]
    }
    if not present_c5_tables and not present_c5_columns:
        return False

    missing_tables = set(C5_TABLE_NAMES) - tables
    missing_columns = {
        (table_name, column_name)
        for table_name, column_names in C5_ADDED_COLUMNS.items()
        for column_name in column_names
        if table_name not in tables or column_name not in columns_by_table[table_name]
    }
    if missing_tables or missing_columns:
        raise C5SchemaCapabilityError('incomplete C.5 persistence schema')
    return True


def has_retained_c5_state(
    bind: Connection | Session,
    *,
    ignore_runtime: bool = False,
    require_complete_schema: bool = True,
) -> bool:
    connection = _bind(bind)
    schema = inspect(connection)
    available_tables = set(schema.get_table_names())
    has_any_c5_shape = bool(set(C5_TABLE_NAMES) & available_tables)
    has_any_c5_shape = has_any_c5_shape or any(
        table_name in available_tables
        and any(
            column['name'] in column_names
            for column in schema.get_columns(table_name)
        )
        for table_name, column_names in C5_ADDED_COLUMNS.items()
    )
    if not has_any_c5_shape:
        return False
    if require_complete_schema:
        c5_schema_available(connection)
    tables = C5_TABLE_NAMES
    if ignore_runtime:
        tables = tuple(
            name for name in tables if name != 'auto_review_runtime_key_states'
        )
    for table_name in tables:
        if table_name not in available_tables:
            continue
        if connection.execute(text(f'SELECT 1 FROM {table_name} LIMIT 1')).first():
            return True

    if 'agent_workflow_threads' in available_tables and connection.execute(
        text(
            'SELECT 1 FROM agent_workflow_threads '
            'WHERE graph_version = :graph_version LIMIT 1'
        ),
        {'graph_version': C5_GRAPH_VERSION},
    ).first():
        return True
    for table_name, column_names in C5_ADDED_COLUMNS.items():
        if table_name not in available_tables:
            continue
        available_columns = {
            column['name'] for column in schema.get_columns(table_name)
        }
        predicates = [
            f'{name} IS NOT NULL'
            for name in column_names
            if name in available_columns
        ]
        if table_name == 'review_items' and 'status' in available_columns:
            predicates.append("status = 'revoked'")
        if not predicates:
            continue
        predicate = ' OR '.join(predicates)
        if connection.execute(
            text(f'SELECT 1 FROM {table_name} WHERE {predicate} LIMIT 1')
        ).first():
            return True
    return False


def _bind(bind: Connection | Session) -> Connection:
    if isinstance(bind, Session):
        return bind.connection()
    return bind
