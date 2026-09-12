"""Approved Task18 legacy-v3 integrity union; no historical backfill."""

import sqlalchemy as sa
from alembic import op

revision = 'b5e6f7a8b9c0'
down_revision = 'a4d5e6f7b8c9'
branch_labels = None
depends_on = None

# Frozen SQL, independent of future application model changes.
OLD_CHECKS = {
    'ck_assistant_messages_dependency_set_schema': "dependency_set_hmac_schema_version IS NULL OR dependency_set_hmac_schema_version = 'assistant-dependency-set-hmac:v2'",
    'ck_assistant_messages_content_origin_xor': "content_write_mode IS NULL OR ((content_write_mode = 'rag_v2_exact' AND content_origin = 'rag_canned' AND evidence_contract_version IS NOT NULL AND evidence_contract_version = 'none-v1' AND serving_dependency_count IS NOT NULL AND serving_dependency_count = 0 AND dependency_set_hmac_schema_version IS NULL AND dependency_set_hmac IS NULL AND parent_selected_evidence_projection_hmac IS NULL AND model_influence_set_hmac IS NULL) OR (content_write_mode = 'rag_v2_exact' AND content_origin = 'rag_assembled' AND evidence_contract_version IS NOT NULL AND evidence_contract_version = 'assistant-evidence:v1' AND serving_dependency_count IS NOT NULL AND serving_dependency_count > 0 AND dependency_set_hmac_schema_version IS NOT NULL AND dependency_set_hmac_schema_version = 'assistant-dependency-set-hmac:v2' AND dependency_set_hmac IS NOT NULL AND length(dependency_set_hmac) = 64 AND parent_selected_evidence_projection_hmac IS NOT NULL AND length(parent_selected_evidence_projection_hmac) = 64 AND model_influence_set_hmac IS NOT NULL AND length(model_influence_set_hmac) = 64) OR (content_write_mode = 'legacy_trimmed' AND content_origin = 'legacy_evidence' AND evidence_contract_version IS NOT NULL AND evidence_contract_version = 'assistant-evidence:v1' AND serving_dependency_count IS NOT NULL AND serving_dependency_count > 0 AND dependency_set_hmac_schema_version IS NOT NULL AND dependency_set_hmac_schema_version = 'assistant-dependency-set-hmac:v2' AND dependency_set_hmac IS NOT NULL AND length(dependency_set_hmac) = 64 AND parent_selected_evidence_projection_hmac IS NOT NULL AND length(parent_selected_evidence_projection_hmac) = 64 AND model_influence_set_hmac IS NULL))",
    'ck_assistant_message_dependency_v2_scope_role': "CASE WHEN dependency_serving_scope IS NULL THEN (dependency_role IS NULL AND dependency_child_hmac IS NULL AND approval_provenance_hmac IS NULL AND evidence_link_set_hmac IS NULL AND legacy_dependency_identity_hmac IS NULL AND model_content_hmac IS NULL AND canonical_citation_projection_hmac IS NULL AND selected_v1_citation_projection_hmac IS NULL AND serving_identity_hmac IS NULL AND serving_version_fingerprint IS NULL AND support_mode IS NULL) ELSE ((dependency_serving_scope = 'rag_v2' AND dependency_role IN ('selected_citation', 'unselected_model_influence') AND dependency_kind IN ('raw_chunk', 'trusted_knowledge')) OR (dependency_serving_scope = 'legacy_v1_only' AND dependency_role = 'selected_citation' AND dependency_kind = 'legacy_unbound')) END",
    'ck_assistant_message_dependency_v2_hmacs': "dependency_serving_scope IS NULL OR (dependency_role IS NOT NULL AND dependency_set_hmac IS NOT NULL AND length(dependency_set_hmac) = 64 AND dependency_child_hmac IS NOT NULL AND length(dependency_child_hmac) = 64 AND fingerprint_key_material_verifier IS NOT NULL AND length(fingerprint_key_material_verifier) = 64 AND model_content_hmac IS NOT NULL AND length(model_content_hmac) = 64 AND canonical_citation_projection_hmac IS NOT NULL AND length(canonical_citation_projection_hmac) = 64 AND (approval_provenance_hmac IS NULL OR length(approval_provenance_hmac) = 64) AND (evidence_link_set_hmac IS NULL OR length(evidence_link_set_hmac) = 64) AND (legacy_dependency_identity_hmac IS NULL OR length(legacy_dependency_identity_hmac) = 64) AND (selected_v1_citation_projection_hmac IS NULL OR length(selected_v1_citation_projection_hmac) = 64) AND (serving_identity_hmac IS NULL OR length(serving_identity_hmac) = 64) AND (serving_version_fingerprint IS NULL OR length(serving_version_fingerprint) = 64) AND ((dependency_role = 'selected_citation' AND selected_v1_citation_projection_hmac IS NOT NULL AND length(selected_v1_citation_projection_hmac) = 64) OR (dependency_role = 'unselected_model_influence' AND selected_v1_citation_projection_hmac IS NULL)))",
    'ck_assistant_message_dependency_v2_support': "dependency_serving_scope IS NULL OR ((dependency_kind = 'raw_chunk' AND support_mode = 'source_observation' AND serving_identity_hmac IS NOT NULL AND serving_version_fingerprint IS NOT NULL AND legacy_dependency_identity_hmac IS NULL) OR (dependency_kind = 'trusted_knowledge' AND support_mode = 'trusted_fact' AND serving_identity_hmac IS NOT NULL AND serving_version_fingerprint IS NOT NULL AND legacy_dependency_identity_hmac IS NULL) OR (dependency_kind = 'legacy_unbound' AND support_mode IS NULL AND serving_identity_hmac IS NULL AND serving_version_fingerprint IS NULL AND legacy_dependency_identity_hmac IS NOT NULL))",
}
NEW_CHECKS = {
    'ck_assistant_messages_content_origin_xor': "content_write_mode IS NULL OR ((content_write_mode = 'rag_v2_exact' AND content_origin = 'rag_canned' AND evidence_contract_version IS NOT NULL AND evidence_contract_version = 'none-v1' AND serving_dependency_count IS NOT NULL AND serving_dependency_count = 0 AND dependency_set_hmac_schema_version IS NULL AND dependency_set_hmac IS NULL AND parent_selected_evidence_projection_hmac IS NULL AND model_influence_set_hmac IS NULL) OR (content_write_mode = 'rag_v2_exact' AND content_origin = 'rag_assembled' AND evidence_contract_version IS NOT NULL AND evidence_contract_version = 'assistant-evidence:v1' AND serving_dependency_count IS NOT NULL AND serving_dependency_count > 0 AND dependency_set_hmac_schema_version IS NOT NULL AND dependency_set_hmac_schema_version = 'assistant-dependency-set-hmac:v2' AND dependency_set_hmac IS NOT NULL AND length(dependency_set_hmac) = 64 AND parent_selected_evidence_projection_hmac IS NOT NULL AND length(parent_selected_evidence_projection_hmac) = 64 AND model_influence_set_hmac IS NOT NULL AND length(model_influence_set_hmac) = 64) OR (content_write_mode = 'legacy_trimmed' AND content_origin = 'legacy_evidence' AND evidence_contract_version IS NOT NULL AND evidence_contract_version = 'assistant-evidence:v1' AND serving_dependency_count IS NOT NULL AND serving_dependency_count > 0 AND dependency_set_hmac_schema_version IS NOT NULL AND dependency_set_hmac_schema_version IN ('assistant-dependency-set-hmac:v2', 'assistant-dependency-set-hmac:v3') AND dependency_set_hmac IS NOT NULL AND length(dependency_set_hmac) = 64 AND parent_selected_evidence_projection_hmac IS NOT NULL AND length(parent_selected_evidence_projection_hmac) = 64 AND model_influence_set_hmac IS NULL))",
    'ck_assistant_messages_dependency_set_schema': "dependency_set_hmac_schema_version IS NULL OR dependency_set_hmac_schema_version IN ('assistant-dependency-set-hmac:v2', 'assistant-dependency-set-hmac:v3')",
    'ck_assistant_message_dependency_v2_scope_role': "CASE WHEN dependency_serving_scope IS NULL THEN (dependency_role IS NULL AND dependency_child_hmac IS NULL AND approval_provenance_hmac IS NULL AND evidence_link_set_hmac IS NULL AND legacy_dependency_identity_hmac IS NULL AND model_content_hmac IS NULL AND canonical_citation_projection_hmac IS NULL AND selected_v1_citation_projection_hmac IS NULL AND serving_identity_hmac IS NULL AND serving_version_fingerprint IS NULL AND support_mode IS NULL) ELSE ((dependency_serving_scope = 'rag_v2' AND dependency_role IN ('selected_citation', 'unselected_model_influence') AND dependency_kind IN ('raw_chunk', 'trusted_knowledge')) OR (dependency_serving_scope = 'legacy_v1_only' AND dependency_role IN ('selected_citation', 'legacy_evidence_influence') AND dependency_kind IN ('raw_chunk', 'trusted_knowledge', 'legacy_unbound'))) END",
    'ck_assistant_message_dependency_legacy_provenance': "dependency_serving_scope IS NULL OR dependency_serving_scope <> 'legacy_v1_only' OR ((approval_link_id IS NOT NULL AND approval_provenance_hmac IS NOT NULL AND evidence_link_set_hmac IS NOT NULL) OR (approval_link_id IS NULL AND approval_provenance_hmac IS NULL AND evidence_link_set_hmac IS NULL))",
    'ck_assistant_message_dependency_v2_hmacs': "dependency_serving_scope IS NULL OR (dependency_role IS NOT NULL AND dependency_set_hmac IS NOT NULL AND length(dependency_set_hmac) = 64 AND dependency_child_hmac IS NOT NULL AND length(dependency_child_hmac) = 64 AND fingerprint_key_material_verifier IS NOT NULL AND length(fingerprint_key_material_verifier) = 64 AND model_content_hmac IS NOT NULL AND length(model_content_hmac) = 64 AND canonical_citation_projection_hmac IS NOT NULL AND length(canonical_citation_projection_hmac) = 64 AND (approval_provenance_hmac IS NULL OR length(approval_provenance_hmac) = 64) AND (evidence_link_set_hmac IS NULL OR length(evidence_link_set_hmac) = 64) AND (legacy_dependency_identity_hmac IS NULL OR length(legacy_dependency_identity_hmac) = 64) AND (selected_v1_citation_projection_hmac IS NULL OR length(selected_v1_citation_projection_hmac) = 64) AND (serving_identity_hmac IS NULL OR length(serving_identity_hmac) = 64) AND (serving_version_fingerprint IS NULL OR length(serving_version_fingerprint) = 64) AND ((dependency_role = 'selected_citation' AND selected_v1_citation_projection_hmac IS NOT NULL AND length(selected_v1_citation_projection_hmac) = 64) OR (dependency_role IN ('unselected_model_influence', 'legacy_evidence_influence') AND selected_v1_citation_projection_hmac IS NULL)))",
    'ck_assistant_message_dependency_v2_support': "dependency_serving_scope IS NULL OR ((dependency_serving_scope = 'rag_v2' AND dependency_kind = 'raw_chunk' AND support_mode = 'source_observation' AND serving_identity_hmac IS NOT NULL AND serving_version_fingerprint IS NOT NULL AND legacy_dependency_identity_hmac IS NULL) OR (dependency_serving_scope = 'rag_v2' AND dependency_kind = 'trusted_knowledge' AND support_mode = 'trusted_fact' AND serving_identity_hmac IS NOT NULL AND serving_version_fingerprint IS NOT NULL AND legacy_dependency_identity_hmac IS NULL) OR (dependency_serving_scope = 'legacy_v1_only' AND support_mode IS NULL AND serving_identity_hmac IS NULL AND serving_version_fingerprint IS NULL AND legacy_dependency_identity_hmac IS NOT NULL))",
}
PG_OLD = "CREATE OR REPLACE FUNCTION rag_validate_assistant_integrity_for(\n          target_message_id bigint\n        ) RETURNS void LANGUAGE plpgsql AS $$\n        DECLARE\n          parent assistant_messages%ROWTYPE;\n          linked agent_runs%ROWTYPE;\n          child_count integer;\n          ordinal_count integer;\n          minimum_ordinal integer;\n          maximum_ordinal integer;\n          selected_count integer;\n          scope_mismatch_count integer;\n          hmac_mismatch_count integer;\n        BEGIN\n          IF target_message_id IS NULL THEN RETURN; END IF;\n          SELECT * INTO parent FROM assistant_messages WHERE id = target_message_id;\n          IF NOT FOUND THEN\n            IF EXISTS (SELECT 1 FROM assistant_message_evidence_dependencies\n                       WHERE assistant_message_id = target_message_id) THEN\n              RAISE EXCEPTION 'Assistant dependencies require a parent message';\n            END IF;\n            RETURN;\n          END IF;\n          IF parent.content_write_mode IS NULL THEN RETURN; END IF;\n          IF parent.content_write_mode = 'rag_v2_exact' THEN\n            SELECT * INTO linked FROM agent_runs WHERE id = parent.linked_agent_run_id;\n            IF NOT FOUND OR\n               linked.run_contract_version IS DISTINCT FROM 'rag-run:v2' OR\n               linked.run_record_phase IS DISTINCT FROM 'final' OR\n               linked.status NOT IN ('complete', 'failed') OR\n               linked.completed_at IS NULL OR\n               linked.metadata ->> 'rag_result_hmac' IS DISTINCT FROM\n                 parent.rag_result_hmac THEN\n              RAISE EXCEPTION 'RAG Assistant message requires linked final rag-run:v2';\n            END IF;\n          END IF;\n          SELECT count(*), count(DISTINCT candidate_ordinal),\n                 min(candidate_ordinal), max(candidate_ordinal),\n                 count(*) FILTER (WHERE dependency_role = 'selected_citation'),\n                 count(*) FILTER (WHERE\n                   (parent.content_origin = 'rag_assembled' AND\n                    dependency_serving_scope IS DISTINCT FROM 'rag_v2') OR\n                   (parent.content_origin = 'legacy_evidence' AND\n                    (dependency_serving_scope IS DISTINCT FROM 'legacy_v1_only' OR\n                     dependency_role IS DISTINCT FROM 'selected_citation'))),\n                 count(*) FILTER (WHERE\n                   dependency_set_hmac IS DISTINCT FROM parent.dependency_set_hmac OR\n                   fingerprint_key_version IS DISTINCT FROM\n                     parent.content_hmac_key_version OR\n                   fingerprint_key_material_verifier IS DISTINCT FROM\n                     parent.content_hmac_key_material_verifier)\n            INTO child_count, ordinal_count, minimum_ordinal, maximum_ordinal,\n                 selected_count, scope_mismatch_count, hmac_mismatch_count\n            FROM assistant_message_evidence_dependencies\n            WHERE assistant_message_id = target_message_id;\n          IF parent.content_origin = 'rag_canned' THEN\n            IF child_count <> 0 OR parent.serving_dependency_count <> 0 THEN\n              RAISE EXCEPTION 'canned Assistant message cannot own dependencies';\n            END IF;\n          ELSIF parent.content_origin IN ('rag_assembled', 'legacy_evidence') THEN\n            IF child_count <> parent.serving_dependency_count OR child_count <= 0 OR\n               ordinal_count <> child_count OR minimum_ordinal <> 0 OR\n               maximum_ordinal <> child_count - 1 OR selected_count <= 0 OR\n               scope_mismatch_count <> 0 OR hmac_mismatch_count <> 0 THEN\n              RAISE EXCEPTION 'Assistant dependency whole-set mismatch';\n            END IF;\n          ELSE\n            RAISE EXCEPTION 'unknown Assistant content origin';\n          END IF;\n        END $$;"
PG_NEW = "CREATE OR REPLACE FUNCTION rag_validate_assistant_integrity_for(\n          target_message_id bigint\n        ) RETURNS void LANGUAGE plpgsql AS $$\n        DECLARE\n          parent assistant_messages%ROWTYPE;\n          linked agent_runs%ROWTYPE;\n          child_count integer;\n          ordinal_count integer;\n          minimum_ordinal integer;\n          maximum_ordinal integer;\n          selected_count integer;\n          scope_mismatch_count integer;\n          hmac_mismatch_count integer;\n        BEGIN\n          IF target_message_id IS NULL THEN RETURN; END IF;\n          SELECT * INTO parent FROM assistant_messages WHERE id = target_message_id;\n          IF NOT FOUND THEN\n            IF EXISTS (SELECT 1 FROM assistant_message_evidence_dependencies\n                       WHERE assistant_message_id = target_message_id) THEN\n              RAISE EXCEPTION 'Assistant dependencies require a parent message';\n            END IF;\n            RETURN;\n          END IF;\n          IF parent.content_write_mode IS NULL THEN RETURN; END IF;\n          IF parent.dependency_set_hmac_schema_version = 'assistant-dependency-set-hmac:v3' THEN\n            IF EXISTS (SELECT 1 FROM assistant_messages p WHERE p.id=target_message_id AND (\n    p.assistant_message_content_hmac IS NULL OR length(p.assistant_message_content_hmac)<>64 OR p.assistant_message_content_hmac !~ '^[0-9a-f]{64}$' OR p.content_origin_hmac IS NULL OR length(p.content_origin_hmac)<>64 OR p.content_origin_hmac !~ '^[0-9a-f]{64}$' OR p.dependency_set_hmac IS NULL OR length(p.dependency_set_hmac)<>64 OR p.dependency_set_hmac !~ '^[0-9a-f]{64}$' OR p.parent_selected_evidence_projection_hmac IS NULL OR length(p.parent_selected_evidence_projection_hmac)<>64 OR p.parent_selected_evidence_projection_hmac !~ '^[0-9a-f]{64}$' OR\n    p.serving_dependency_count IS NULL OR p.serving_dependency_count <= 0 OR\n    (SELECT count(*) FROM assistant_message_evidence_dependencies d WHERE d.assistant_message_id=p.id ) <> p.serving_dependency_count OR\n    (SELECT min(candidate_ordinal) FROM assistant_message_evidence_dependencies WHERE assistant_message_id=p.id) <> 0 OR\n    (SELECT max(candidate_ordinal) FROM assistant_message_evidence_dependencies WHERE assistant_message_id=p.id) <> p.serving_dependency_count-1 OR\n    (SELECT count(*) FROM assistant_message_evidence_dependencies d WHERE d.assistant_message_id=p.id AND d.dependency_role='selected_citation') <> jsonb_array_length(p.citations::jsonb) OR\n    ((SELECT count(*) FROM assistant_message_evidence_dependencies d WHERE d.assistant_message_id=p.id AND d.dependency_role='selected_citation')=0 AND (NOT COALESCE((p.metadata::jsonb -> 'evidence_derived' = 'true'::jsonb), false) OR\n       jsonb_array_length(p.source_ids::jsonb)<>0 OR jsonb_array_length(p.source_links::jsonb)<>0 OR jsonb_array_length(p.source_snippets::jsonb)<>0)) OR\n    EXISTS (SELECT 1 FROM assistant_message_evidence_dependencies d WHERE d.assistant_message_id=p.id AND (\n       d.dependency_serving_scope IS DISTINCT FROM 'legacy_v1_only' OR\n       d.dependency_role NOT IN ('selected_citation','legacy_evidence_influence') OR\n       d.dependency_set_hmac IS DISTINCT FROM p.dependency_set_hmac OR\n       d.fingerprint_key_version IS DISTINCT FROM p.content_hmac_key_version OR\n       d.fingerprint_key_material_verifier IS DISTINCT FROM p.content_hmac_key_material_verifier OR\n       d.serving_identity_hmac IS NOT NULL OR d.serving_version_fingerprint IS NOT NULL OR d.support_mode IS NOT NULL OR\n       d.dependency_set_hmac IS NULL OR length(d.dependency_set_hmac)<>64 OR d.dependency_set_hmac !~ '^[0-9a-f]{64}$' OR d.dependency_child_hmac IS NULL OR length(d.dependency_child_hmac)<>64 OR d.dependency_child_hmac !~ '^[0-9a-f]{64}$' OR d.model_content_hmac IS NULL OR length(d.model_content_hmac)<>64 OR d.model_content_hmac !~ '^[0-9a-f]{64}$' OR d.canonical_citation_projection_hmac IS NULL OR length(d.canonical_citation_projection_hmac)<>64 OR d.canonical_citation_projection_hmac !~ '^[0-9a-f]{64}$' OR d.legacy_dependency_identity_hmac IS NULL OR length(d.legacy_dependency_identity_hmac)<>64 OR d.legacy_dependency_identity_hmac !~ '^[0-9a-f]{64}$' OR (d.selected_v1_citation_projection_hmac IS NOT NULL AND (length(d.selected_v1_citation_projection_hmac)<>64 OR d.selected_v1_citation_projection_hmac !~ '^[0-9a-f]{64}$')) OR (d.approval_provenance_hmac IS NOT NULL AND (length(d.approval_provenance_hmac)<>64 OR d.approval_provenance_hmac !~ '^[0-9a-f]{64}$')) OR (d.evidence_link_set_hmac IS NOT NULL AND (length(d.evidence_link_set_hmac)<>64 OR d.evidence_link_set_hmac !~ '^[0-9a-f]{64}$')) OR\n       (d.approval_link_id IS NOT NULL AND (\n          NOT EXISTS (SELECT 1 FROM assistant_message_knowledge_evidence_refs r WHERE r.dependency_id=d.id) OR\n          (SELECT count(*) FROM assistant_message_knowledge_evidence_refs r WHERE r.dependency_id=d.id) <>\n          (SELECT count(*) FROM trusted_knowledge_evidence_links e WHERE e.approval_link_id=d.approval_link_id))) OR\n       EXISTS (SELECT 1 FROM assistant_message_knowledge_evidence_refs r\n          LEFT JOIN trusted_knowledge_evidence_links e ON e.id=r.trusted_knowledge_evidence_link_id\n          WHERE r.dependency_id=d.id AND (r.assistant_message_id IS DISTINCT FROM p.id OR\n          r.approval_link_id IS DISTINCT FROM d.approval_link_id OR e.id IS NULL OR\n          e.approval_link_id IS DISTINCT FROM d.approval_link_id))\n    ))\n    )) THEN\n              RAISE EXCEPTION 'legacy v3 published integrity mismatch';\n            END IF;\n            RETURN;\n          END IF;\n          IF parent.content_write_mode = 'rag_v2_exact' THEN\n            SELECT * INTO linked FROM agent_runs WHERE id = parent.linked_agent_run_id;\n            IF NOT FOUND OR\n               linked.run_contract_version IS DISTINCT FROM 'rag-run:v2' OR\n               linked.run_record_phase IS DISTINCT FROM 'final' OR\n               linked.status NOT IN ('complete', 'failed') OR\n               linked.completed_at IS NULL OR\n               linked.metadata ->> 'rag_result_hmac' IS DISTINCT FROM\n                 parent.rag_result_hmac THEN\n              RAISE EXCEPTION 'RAG Assistant message requires linked final rag-run:v2';\n            END IF;\n          END IF;\n          SELECT count(*), count(DISTINCT candidate_ordinal),\n                 min(candidate_ordinal), max(candidate_ordinal),\n                 count(*) FILTER (WHERE dependency_role = 'selected_citation'),\n                 count(*) FILTER (WHERE\n                   (parent.content_origin = 'rag_assembled' AND\n                    dependency_serving_scope IS DISTINCT FROM 'rag_v2') OR\n                   (parent.content_origin = 'legacy_evidence' AND\n                    (dependency_serving_scope IS DISTINCT FROM 'legacy_v1_only' OR\n                     dependency_role IS DISTINCT FROM 'selected_citation' OR dependency_kind IS DISTINCT FROM 'legacy_unbound'))),\n                 count(*) FILTER (WHERE\n                   dependency_set_hmac IS DISTINCT FROM parent.dependency_set_hmac OR\n                   fingerprint_key_version IS DISTINCT FROM\n                     parent.content_hmac_key_version OR\n                   fingerprint_key_material_verifier IS DISTINCT FROM\n                     parent.content_hmac_key_material_verifier)\n            INTO child_count, ordinal_count, minimum_ordinal, maximum_ordinal,\n                 selected_count, scope_mismatch_count, hmac_mismatch_count\n            FROM assistant_message_evidence_dependencies\n            WHERE assistant_message_id = target_message_id;\n          IF parent.content_origin = 'rag_canned' THEN\n            IF child_count <> 0 OR parent.serving_dependency_count <> 0 THEN\n              RAISE EXCEPTION 'canned Assistant message cannot own dependencies';\n            END IF;\n          ELSIF parent.content_origin IN ('rag_assembled', 'legacy_evidence') THEN\n            IF child_count <> parent.serving_dependency_count OR child_count <= 0 OR\n               ordinal_count <> child_count OR minimum_ordinal <> 0 OR\n               maximum_ordinal <> child_count - 1 OR selected_count <= 0 OR\n               scope_mismatch_count <> 0 OR hmac_mismatch_count <> 0 THEN\n              RAISE EXCEPTION 'Assistant dependency whole-set mismatch';\n            END IF;\n          ELSE\n            RAISE EXCEPTION 'unknown Assistant content origin';\n          END IF;\n        END $$;"
SQLITE_QUERY = "SELECT 1 FROM assistant_messages p WHERE p.id={target} AND ((p.dependency_set_hmac_schema_version='assistant-dependency-set-hmac:v3' AND (\n    p.assistant_message_content_hmac IS NULL OR length(p.assistant_message_content_hmac)<>64 OR p.assistant_message_content_hmac GLOB '*[^0-9a-f]*' OR p.content_origin_hmac IS NULL OR length(p.content_origin_hmac)<>64 OR p.content_origin_hmac GLOB '*[^0-9a-f]*' OR p.dependency_set_hmac IS NULL OR length(p.dependency_set_hmac)<>64 OR p.dependency_set_hmac GLOB '*[^0-9a-f]*' OR p.parent_selected_evidence_projection_hmac IS NULL OR length(p.parent_selected_evidence_projection_hmac)<>64 OR p.parent_selected_evidence_projection_hmac GLOB '*[^0-9a-f]*' OR\n    p.serving_dependency_count IS NULL OR p.serving_dependency_count <= 0 OR\n    (SELECT count(*) FROM assistant_message_evidence_dependencies d WHERE d.assistant_message_id=p.id ) <> p.serving_dependency_count OR\n    (SELECT min(candidate_ordinal) FROM assistant_message_evidence_dependencies WHERE assistant_message_id=p.id) <> 0 OR\n    (SELECT max(candidate_ordinal) FROM assistant_message_evidence_dependencies WHERE assistant_message_id=p.id) <> p.serving_dependency_count-1 OR\n    (SELECT count(*) FROM assistant_message_evidence_dependencies d WHERE d.assistant_message_id=p.id AND d.dependency_role='selected_citation') <> json_array_length(p.citations) OR\n    ((SELECT count(*) FROM assistant_message_evidence_dependencies d WHERE d.assistant_message_id=p.id AND d.dependency_role='selected_citation')=0 AND (NOT COALESCE((json_type(p.metadata, '$.evidence_derived') = 'true'), false) OR\n       json_array_length(p.source_ids)<>0 OR json_array_length(p.source_links)<>0 OR json_array_length(p.source_snippets)<>0)) OR\n    EXISTS (SELECT 1 FROM assistant_message_evidence_dependencies d WHERE d.assistant_message_id=p.id AND (\n       d.dependency_serving_scope IS DISTINCT FROM 'legacy_v1_only' OR\n       d.dependency_role NOT IN ('selected_citation','legacy_evidence_influence') OR\n       d.dependency_set_hmac IS DISTINCT FROM p.dependency_set_hmac OR\n       d.fingerprint_key_version IS DISTINCT FROM p.content_hmac_key_version OR\n       d.fingerprint_key_material_verifier IS DISTINCT FROM p.content_hmac_key_material_verifier OR\n       d.serving_identity_hmac IS NOT NULL OR d.serving_version_fingerprint IS NOT NULL OR d.support_mode IS NOT NULL OR\n       d.dependency_set_hmac IS NULL OR length(d.dependency_set_hmac)<>64 OR d.dependency_set_hmac GLOB '*[^0-9a-f]*' OR d.dependency_child_hmac IS NULL OR length(d.dependency_child_hmac)<>64 OR d.dependency_child_hmac GLOB '*[^0-9a-f]*' OR d.model_content_hmac IS NULL OR length(d.model_content_hmac)<>64 OR d.model_content_hmac GLOB '*[^0-9a-f]*' OR d.canonical_citation_projection_hmac IS NULL OR length(d.canonical_citation_projection_hmac)<>64 OR d.canonical_citation_projection_hmac GLOB '*[^0-9a-f]*' OR d.legacy_dependency_identity_hmac IS NULL OR length(d.legacy_dependency_identity_hmac)<>64 OR d.legacy_dependency_identity_hmac GLOB '*[^0-9a-f]*' OR (d.selected_v1_citation_projection_hmac IS NOT NULL AND (length(d.selected_v1_citation_projection_hmac)<>64 OR d.selected_v1_citation_projection_hmac GLOB '*[^0-9a-f]*')) OR (d.approval_provenance_hmac IS NOT NULL AND (length(d.approval_provenance_hmac)<>64 OR d.approval_provenance_hmac GLOB '*[^0-9a-f]*')) OR (d.evidence_link_set_hmac IS NOT NULL AND (length(d.evidence_link_set_hmac)<>64 OR d.evidence_link_set_hmac GLOB '*[^0-9a-f]*')) OR\n       (d.approval_link_id IS NOT NULL AND (\n          NOT EXISTS (SELECT 1 FROM assistant_message_knowledge_evidence_refs r WHERE r.dependency_id=d.id) OR\n          (SELECT count(*) FROM assistant_message_knowledge_evidence_refs r WHERE r.dependency_id=d.id) <>\n          (SELECT count(*) FROM trusted_knowledge_evidence_links e WHERE e.approval_link_id=d.approval_link_id))) OR\n       EXISTS (SELECT 1 FROM assistant_message_knowledge_evidence_refs r\n          LEFT JOIN trusted_knowledge_evidence_links e ON e.id=r.trusted_knowledge_evidence_link_id\n          WHERE r.dependency_id=d.id AND (r.assistant_message_id IS DISTINCT FROM p.id OR\n          r.approval_link_id IS DISTINCT FROM d.approval_link_id OR e.id IS NULL OR\n          e.approval_link_id IS DISTINCT FROM d.approval_link_id))\n    ))\n    )) OR (p.dependency_set_hmac_schema_version='assistant-dependency-set-hmac:v2' AND p.content_origin='legacy_evidence' AND EXISTS (SELECT 1 FROM assistant_message_evidence_dependencies d WHERE d.assistant_message_id=p.id AND (d.dependency_kind<>'legacy_unbound' OR d.dependency_role<>'selected_citation' OR d.dependency_serving_scope<>'legacy_v1_only'))))"


PG_EXACTNESS_OLD = """        CREATE OR REPLACE FUNCTION enforce_assistant_dependency_exactness()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE new_row jsonb := COALESCE(to_jsonb(NEW), '{}'::jsonb);
        DECLARE old_row jsonb := COALESCE(to_jsonb(OLD), '{}'::jsonb);
        DECLARE dependency_id integer;
        DECLARE dep assistant_message_evidence_dependencies%ROWTYPE;
        DECLARE expected_count integer;
        DECLARE actual_count integer;
        DECLARE human_proven boolean := false;
        BEGIN
          dependency_id := COALESCE(
            (new_row ->> 'dependency_id')::integer,
            (old_row ->> 'dependency_id')::integer,
            (new_row ->> 'id')::integer,
            (old_row ->> 'id')::integer);
          SELECT * INTO dep FROM assistant_message_evidence_dependencies
          WHERE id = dependency_id;
          IF NOT FOUND THEN RETURN COALESCE(NEW, OLD); END IF;
          IF dep.dependency_kind = 'raw_chunk' THEN
            IF NOT EXISTS (
              SELECT 1 FROM document_chunks chunk
              JOIN document_parser_runs parser ON parser.id = chunk.parser_run_id
              JOIN document_versions version ON version.id = chunk.version_id
              JOIN documents document ON document.id = version.document_id
              JOIN sources source ON source.id = chunk.source_id
              WHERE chunk.id = dep.document_chunk_id
                AND chunk.version_id = dep.document_version_id
                AND chunk.source_id = dep.source_id
                AND chunk.parser_run_id = dep.parser_run_id
                AND document.current_document_version_id = dep.document_version_id
                AND dep.current_document_version_id = dep.document_version_id
                AND parser.document_id = document.id
                AND parser.document_version_id = dep.document_version_id
                AND parser.source_id = dep.source_id
                AND parser.server_content_signature_schema =
                    dep.server_content_signature_schema
                AND parser.server_content_signature = dep.server_content_signature
                AND parser.parser_policy_version = dep.parser_policy_version
                AND parser.parser_version = dep.parser_version
                AND parser.chunk_policy_version = dep.chunk_policy_version
                AND source.server_content_signature_schema =
                    dep.server_content_signature_schema
                AND source.server_content_signature = dep.server_content_signature
            ) THEN RAISE EXCEPTION 'raw dependency is not current'; END IF;
          ELSIF dep.approval_link_id IS NOT NULL THEN
            SELECT COUNT(*) INTO expected_count FROM trusted_knowledge_evidence_links
            WHERE approval_link_id = dep.approval_link_id;
            SELECT COUNT(*) INTO actual_count
            FROM assistant_message_knowledge_evidence_refs ref
            WHERE ref.dependency_id = dep.id;
            IF expected_count < 1 OR actual_count <> expected_count OR NOT EXISTS (
              SELECT 1 FROM trusted_knowledge_approval_links approval
              WHERE approval.id = dep.approval_link_id
                AND approval.knowledge_type = dep.knowledge_type
                AND approval.knowledge_id = dep.knowledge_id
                AND approval.active = true AND approval.revoked_at IS NULL
            ) THEN RAISE EXCEPTION 'complete active approval effect required'; END IF;
          ELSE
            SELECT EXISTS (
              SELECT 1 FROM review_items item
              WHERE item.id = dep.legacy_source_review_item_id
                AND item.status = 'approved' AND item.resolution_source = 'human'
            ) INTO human_proven;
            IF human_proven AND dep.knowledge_type = 'timeline_event' THEN
              SELECT EXISTS (SELECT 1 FROM timeline_events target
                WHERE target.id = dep.knowledge_id AND
                  target.source_review_item_id = dep.legacy_source_review_item_id)
                INTO human_proven;
            ELSIF human_proven AND dep.knowledge_type = 'history_event' THEN
              SELECT EXISTS (SELECT 1 FROM history_events target
                WHERE target.id = dep.knowledge_id AND
                  target.source_review_item_id = dep.legacy_source_review_item_id)
                INTO human_proven;
            ELSIF human_proven AND dep.knowledge_type = 'decision_record' THEN
              SELECT EXISTS (SELECT 1 FROM decision_records target
                WHERE target.id = dep.knowledge_id AND
                  target.source_review_item_id = dep.legacy_source_review_item_id)
                INTO human_proven;
            ELSIF human_proven AND dep.knowledge_type = 'todo' THEN
              SELECT EXISTS (SELECT 1 FROM todos target
                WHERE target.id = dep.knowledge_id AND
                  target.source_review_item_id = dep.legacy_source_review_item_id)
                INTO human_proven;
            ELSE human_proven := false;
            END IF;
            IF NOT human_proven THEN RAISE EXCEPTION 'legacy human proof required';
            END IF;
          END IF;
          RETURN COALESCE(NEW, OLD);
        END $$"""


PG_EXACTNESS_NEW = """        CREATE OR REPLACE FUNCTION enforce_assistant_dependency_exactness()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE new_row jsonb := COALESCE(to_jsonb(NEW), '{}'::jsonb);
        DECLARE old_row jsonb := COALESCE(to_jsonb(OLD), '{}'::jsonb);
        DECLARE dependency_id integer;
        DECLARE dep assistant_message_evidence_dependencies%ROWTYPE;
        DECLARE expected_count integer;
        DECLARE actual_count integer;
        DECLARE human_proven boolean := false;
        BEGIN
          dependency_id := COALESCE(
            (new_row ->> 'dependency_id')::integer,
            (old_row ->> 'dependency_id')::integer,
            (new_row ->> 'id')::integer,
            (old_row ->> 'id')::integer);
          SELECT * INTO dep FROM assistant_message_evidence_dependencies
          WHERE id = dependency_id;
          IF NOT FOUND THEN RETURN COALESCE(NEW, OLD); END IF;
          IF dep.dependency_serving_scope = 'legacy_v1_only'
             AND dep.approval_link_id IS NULL
             AND dep.legacy_source_review_item_id IS NULL
             AND (dep.dependency_kind = 'legacy_unbound' OR
                  (dep.dependency_kind = 'trusted_knowledge' AND dep.legacy_human_base))
             AND EXISTS (SELECT 1 FROM assistant_messages p WHERE p.id=dep.assistant_message_id
                 AND p.dependency_set_hmac_schema_version='assistant-dependency-set-hmac:v3') THEN
            IF NOT EXISTS (
              SELECT 1 FROM (
                SELECT 'decision_record' AS kind, id, source_review_item_id, review_status FROM decision_records
                UNION ALL SELECT 'history_event', id, source_review_item_id, review_status FROM history_events
                UNION ALL SELECT 'timeline_event', id, source_review_item_id, review_status FROM timeline_events
                UNION ALL SELECT 'todo', id, source_review_item_id, review_status FROM todos
              ) target
              WHERE dep.serving_document_id IN (target.kind || ':' || target.id::text,
                   CASE WHEN target.kind='decision_record' THEN 'decision:' || target.id::text ELSE NULL END)
                AND target.source_review_item_id IS NULL AND target.review_status='approved'
                AND ((dep.dependency_kind='legacy_unbound' AND NOT EXISTS (
                     SELECT 1 FROM trusted_knowledge_approval_links a WHERE a.knowledge_id=target.id
                     AND a.knowledge_type IN (target.kind, CASE WHEN target.kind='decision_record' THEN 'decision' ELSE target.kind END)))
                     OR (dep.knowledge_id=target.id AND dep.knowledge_type IN (
                         target.kind, CASE WHEN target.kind='decision_record' THEN 'decision' ELSE target.kind END)))
            ) OR EXISTS (SELECT 1 FROM assistant_message_knowledge_evidence_refs r WHERE r.dependency_id=dep.id)
            THEN RAISE EXCEPTION 'genuine legacy authority required'; END IF;
            RETURN COALESCE(NEW, OLD);
          END IF;
          IF dep.dependency_kind = 'raw_chunk' THEN
            IF NOT EXISTS (
              SELECT 1 FROM document_chunks chunk
              JOIN document_parser_runs parser ON parser.id = chunk.parser_run_id
              JOIN document_versions version ON version.id = chunk.version_id
              JOIN documents document ON document.id = version.document_id
              JOIN sources source ON source.id = chunk.source_id
              WHERE chunk.id = dep.document_chunk_id
                AND chunk.version_id = dep.document_version_id
                AND chunk.source_id = dep.source_id
                AND chunk.parser_run_id = dep.parser_run_id
                AND document.current_document_version_id = dep.document_version_id
                AND dep.current_document_version_id = dep.document_version_id
                AND parser.document_id = document.id
                AND parser.document_version_id = dep.document_version_id
                AND parser.source_id = dep.source_id
                AND parser.server_content_signature_schema =
                    dep.server_content_signature_schema
                AND parser.server_content_signature = dep.server_content_signature
                AND parser.parser_policy_version = dep.parser_policy_version
                AND parser.parser_version = dep.parser_version
                AND parser.chunk_policy_version = dep.chunk_policy_version
                AND source.server_content_signature_schema =
                    dep.server_content_signature_schema
                AND source.server_content_signature = dep.server_content_signature
            ) THEN RAISE EXCEPTION 'raw dependency is not current'; END IF;
          ELSIF dep.approval_link_id IS NOT NULL THEN
            SELECT COUNT(*) INTO expected_count FROM trusted_knowledge_evidence_links
            WHERE approval_link_id = dep.approval_link_id;
            SELECT COUNT(*) INTO actual_count
            FROM assistant_message_knowledge_evidence_refs ref
            WHERE ref.dependency_id = dep.id;
            IF expected_count < 1 OR actual_count <> expected_count OR NOT EXISTS (
              SELECT 1 FROM trusted_knowledge_approval_links approval
              WHERE approval.id = dep.approval_link_id
                AND approval.knowledge_type = dep.knowledge_type
                AND approval.knowledge_id = dep.knowledge_id
                AND approval.active = true AND approval.revoked_at IS NULL
            ) THEN RAISE EXCEPTION 'complete active approval effect required'; END IF;
          ELSE
            SELECT EXISTS (
              SELECT 1 FROM review_items item
              WHERE item.id = dep.legacy_source_review_item_id
                AND item.status = 'approved' AND item.resolution_source = 'human'
            ) INTO human_proven;
            IF human_proven AND dep.knowledge_type = 'timeline_event' THEN
              SELECT EXISTS (SELECT 1 FROM timeline_events target
                WHERE target.id = dep.knowledge_id AND
                  target.source_review_item_id = dep.legacy_source_review_item_id)
                INTO human_proven;
            ELSIF human_proven AND dep.knowledge_type = 'history_event' THEN
              SELECT EXISTS (SELECT 1 FROM history_events target
                WHERE target.id = dep.knowledge_id AND
                  target.source_review_item_id = dep.legacy_source_review_item_id)
                INTO human_proven;
            ELSIF human_proven AND dep.knowledge_type = 'decision_record' THEN
              SELECT EXISTS (SELECT 1 FROM decision_records target
                WHERE target.id = dep.knowledge_id AND
                  target.source_review_item_id = dep.legacy_source_review_item_id)
                INTO human_proven;
            ELSIF human_proven AND dep.knowledge_type = 'todo' THEN
              SELECT EXISTS (SELECT 1 FROM todos target
                WHERE target.id = dep.knowledge_id AND
                  target.source_review_item_id = dep.legacy_source_review_item_id)
                INTO human_proven;
            ELSE human_proven := false;
            END IF;
            IF NOT human_proven THEN RAISE EXCEPTION 'legacy human proof required';
            END IF;
          END IF;
          RETURN COALESCE(NEW, OLD);
        END $$"""


PG_STAGING = """CREATE OR REPLACE FUNCTION enforce_assistant_legacy_v3_staging()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE mutable_columns text[] := ARRAY[
 'dependency_set_hmac', 'dependency_serving_scope', 'dependency_role',
 'dependency_child_hmac', 'legacy_dependency_identity_hmac', 'model_content_hmac',
 'canonical_citation_projection_hmac', 'selected_v1_citation_projection_hmac',
 'approval_provenance_hmac', 'evidence_link_set_hmac'];
BEGIN
 IF TG_OP='UPDATE' AND OLD.dependency_serving_scope IS NULL
    AND NEW.dependency_serving_scope='legacy_v1_only'
    AND (to_jsonb(NEW)-mutable_columns)=(to_jsonb(OLD)-mutable_columns)
    AND EXISTS (SELECT 1 FROM assistant_messages p
        WHERE p.id=OLD.assistant_message_id AND p.content_write_mode IS NULL
        AND p.xmin::text=(txid_current() % 4294967296)::text)
    AND EXISTS (SELECT 1 FROM assistant_message_evidence_dependencies d WHERE d.id=OLD.id
        AND d.xmin::text=(txid_current() % 4294967296)::text)
 THEN RETURN NEW; END IF;
 RAISE EXCEPTION 'C.5 audit/evidence row is append-only';
END $$;"""


def _sqlite_statements():
    tables = {
        'assistant_messages': ('NEW.id', 'OLD.id'),
        'assistant_message_evidence_dependencies': (
            'NEW.assistant_message_id',
            'OLD.assistant_message_id',
        ),
        'assistant_message_knowledge_evidence_refs': (
            'NEW.assistant_message_id',
            'OLD.assistant_message_id',
        ),
    }
    for table, (new, old) in tables.items():
        for action in ('INSERT', 'UPDATE', 'DELETE'):
            targets = (
                [new]
                if action == 'INSERT'
                else [old]
                if action == 'DELETE'
                else [old, new]
            )
            conditions = ' OR '.join(
                f'EXISTS ({SQLITE_QUERY.format(target=target)})' for target in targets
            )
            if table == 'assistant_messages' and action == 'UPDATE':
                conditions += " OR (OLD.dependency_set_hmac_schema_version='assistant-dependency-set-hmac:v3' AND NEW.dependency_set_hmac_schema_version IS DISTINCT FROM 'assistant-dependency-set-hmac:v3')"
            yield f"CREATE TRIGGER IF NOT EXISTS legacy_v3_{table}_{action.lower()} AFTER {action} ON {table} BEGIN SELECT CASE WHEN {conditions} THEN RAISE(ABORT,'legacy v3 published integrity mismatch') END; END"


def _checks(values, previous):
    for table, prefix in (
        ('assistant_messages', 'ck_assistant_messages_'),
        ('assistant_message_evidence_dependencies', 'ck_assistant_message_dependency_'),
    ):
        with op.batch_alter_table(table) as batch:
            for name in previous:
                if name.startswith(prefix):
                    batch.drop_constraint(name, type_='check')
            for name, sql in values.items():
                if name.startswith(prefix):
                    batch.create_check_constraint(name, sql)


def _drop_sqlite_guards():
    for table in (
        'assistant_messages',
        'assistant_message_evidence_dependencies',
        'assistant_message_knowledge_evidence_refs',
    ):
        for action in ('insert', 'update', 'delete'):
            op.execute(sa.text(f'DROP TRIGGER IF EXISTS legacy_v3_{table}_{action}'))


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name == 'sqlite':
        _drop_sqlite_guards()
    _checks(NEW_CHECKS, OLD_CHECKS)
    if bind.dialect.name == 'sqlite':
        for statement in _sqlite_statements():
            op.execute(sa.text(statement))
    elif bind.dialect.name == 'postgresql':
        op.execute(sa.text(PG_NEW))
        op.execute(
            sa.text("""
          CREATE CONSTRAINT TRIGGER rag_assistant_integrity_guard_refs
          AFTER INSERT OR UPDATE OR DELETE ON assistant_message_knowledge_evidence_refs
          DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
          EXECUTE FUNCTION rag_validate_assistant_integrity();
        """)
        )

        op.execute(sa.text(PG_STAGING))
        op.execute(sa.text(PG_EXACTNESS_NEW))
        op.execute(
            sa.text("""
          CREATE FUNCTION enforce_assistant_legacy_v3_marker() RETURNS trigger LANGUAGE plpgsql AS $$
          BEGIN
            IF OLD.dependency_set_hmac_schema_version='assistant-dependency-set-hmac:v3'
               AND NEW.dependency_set_hmac_schema_version IS DISTINCT FROM 'assistant-dependency-set-hmac:v3'
            THEN RAISE EXCEPTION 'published legacy v3 marker is immutable'; END IF;
            RETURN NEW;
          END $$;
          CREATE TRIGGER legacy_v3_parent_marker BEFORE UPDATE ON assistant_messages
          FOR EACH ROW EXECUTE FUNCTION enforce_assistant_legacy_v3_marker();
        """)
        )
        op.execute(
            sa.text(
                'DROP TRIGGER trg_assistant_dependency_immutable ON assistant_message_evidence_dependencies'
            )
        )
        op.execute(
            sa.text(
                'CREATE TRIGGER trg_assistant_dependency_immutable BEFORE UPDATE OR DELETE ON assistant_message_evidence_dependencies FOR EACH ROW EXECUTE FUNCTION enforce_assistant_legacy_v3_staging()'
            )
        )


def downgrade():
    bind = op.get_bind()
    if bind.scalar(
        sa.text(
            "SELECT count(*) FROM assistant_messages WHERE dependency_set_hmac_schema_version='assistant-dependency-set-hmac:v3'"
        )
    ):
        raise ValueError('cannot downgrade while legacy v3 messages exist')
    if bind.dialect.name == 'sqlite':
        _drop_sqlite_guards()
    elif bind.dialect.name == 'postgresql':
        op.execute(
            sa.text(
                'DROP TRIGGER IF EXISTS rag_assistant_integrity_guard_refs ON assistant_message_knowledge_evidence_refs'
            )
        )
        op.execute(sa.text(PG_OLD))
        op.execute(sa.text(PG_EXACTNESS_OLD))
        op.execute(
            sa.text('DROP TRIGGER legacy_v3_parent_marker ON assistant_messages')
        )
        op.execute(sa.text('DROP FUNCTION enforce_assistant_legacy_v3_marker()'))
        op.execute(
            sa.text(
                'DROP TRIGGER trg_assistant_dependency_immutable ON assistant_message_evidence_dependencies'
            )
        )
        op.execute(
            sa.text(
                'CREATE TRIGGER trg_assistant_dependency_immutable BEFORE UPDATE OR DELETE ON assistant_message_evidence_dependencies FOR EACH ROW EXECUTE FUNCTION reject_c5_immutable_mutation()'
            )
        )
        op.execute(sa.text('DROP FUNCTION enforce_assistant_legacy_v3_staging()'))
    _checks(OLD_CHECKS, NEW_CHECKS)
