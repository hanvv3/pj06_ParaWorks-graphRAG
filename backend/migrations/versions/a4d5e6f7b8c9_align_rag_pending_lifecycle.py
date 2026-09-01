"""Align RAG pending-projection lifecycle with the frozen null-outcome contract."""

from __future__ import annotations

from alembic import op
from sqlalchemy import text

revision = 'a4d5e6f7b8c9'
down_revision = 'f3c4d5e6a7b8'
branch_labels = None
depends_on = None


def _guard(*, pending_requires_null: bool, admission_requires_abandoned: bool) -> str:
    pending_outcome = 'IS NOT NULL' if pending_requires_null else 'IS NULL'
    admission_abandoned = (
        'abandoned_count = 0 OR ' if admission_requires_abandoned else ''
    )
    return f"""
    CREATE OR REPLACE FUNCTION rag_validate_agent_run_v2_costs_for(target_run_id bigint)
    RETURNS void LANGUAGE plpgsql AS $$
    DECLARE
      parent_version text; parent_phase text; parent_status text;
      parent_outcome text; parent_completed_at timestamptz;
      parent_projection_fence text; parent_total numeric;
      child_count integer; child_components integer; child_total numeric;
      not_attempted_count integer; dispatching_count integer;
      terminal_count integer; abandoned_count integer;
    BEGIN
      IF target_run_id IS NULL THEN RETURN; END IF;
      SELECT run_contract_version, run_record_phase, status,
             metadata ->> 'outcome', completed_at,
             projection_owner_fence_hmac, total_charged_cost_usd
        INTO parent_version, parent_phase, parent_status, parent_outcome,
             parent_completed_at, parent_projection_fence, parent_total
        FROM agent_runs WHERE id = target_run_id;
      SELECT count(*), count(DISTINCT component),
             COALESCE(sum(charged_cost_usd), 0.000000),
             count(*) FILTER (WHERE dispatch_state = 'not_attempted'),
             count(*) FILTER (WHERE dispatch_state = 'dispatching'),
             count(*) FILTER (WHERE dispatch_state = 'terminal'),
             count(*) FILTER (WHERE dispatch_state = 'abandoned_unknown')
        INTO child_count, child_components, child_total,
             not_attempted_count, dispatching_count, terminal_count,
             abandoned_count
        FROM agent_run_cost_components WHERE agent_run_id = target_run_id;
      IF parent_version IS NULL THEN
        IF child_count <> 0 THEN
          RAISE EXCEPTION 'legacy or missing AgentRun cannot own D cost children';
        END IF;
        RETURN;
      END IF;
      IF parent_version <> 'rag-run:v2' OR parent_total IS NULL OR
         child_count <> 2 OR child_components <> 2 OR
         NOT EXISTS (SELECT 1 FROM agent_run_cost_components
                     WHERE agent_run_id = target_run_id
                       AND component = 'query_embedding' AND component_ordinal = 0) OR
         NOT EXISTS (SELECT 1 FROM agent_run_cost_components
                     WHERE agent_run_id = target_run_id
                       AND component = 'answer_generation' AND component_ordinal = 1) OR
         child_total <> parent_total THEN
        RAISE EXCEPTION 'rag-run:v2 requires exact two balanced cost children';
      END IF;
      IF parent_phase = 'admission' THEN
        IF parent_status <> 'running' OR parent_outcome IS NOT NULL OR
           parent_completed_at IS NOT NULL OR parent_projection_fence IS NOT NULL OR
           abandoned_count <> 0 OR terminal_count = 2 THEN
          RAISE EXCEPTION 'invalid rag-run:v2 admission lifecycle';
        END IF;
      ELSIF parent_phase = 'cost_finalized_pending_projection' THEN
        IF parent_status <> 'running' OR parent_outcome {pending_outcome} OR
           parent_completed_at IS NOT NULL OR parent_projection_fence IS NULL OR
           terminal_count <> 2 THEN
          RAISE EXCEPTION 'invalid rag-run:v2 pending-projection lifecycle';
        END IF;
      ELSIF parent_phase = 'final' THEN
        IF parent_status NOT IN ('complete', 'failed') OR parent_outcome IS NULL OR
           parent_completed_at IS NULL OR terminal_count <> 2 OR
           not_attempted_count <> 0 OR dispatching_count <> 0 OR abandoned_count <> 0 THEN
          RAISE EXCEPTION 'invalid rag-run:v2 final lifecycle';
        END IF;
      ELSIF parent_phase = 'admission_only' THEN
        IF parent_status <> 'failed' OR parent_outcome <> 'abandoned_unknown' OR
           parent_completed_at IS NULL OR parent_projection_fence IS NOT NULL OR
           not_attempted_count <> 0 OR dispatching_count <> 0 OR
           {admission_abandoned}terminal_count + abandoned_count <> 2 THEN
          RAISE EXCEPTION 'invalid rag-run:v2 admission-only lifecycle';
        END IF;
      ELSE
        RAISE EXCEPTION 'invalid rag-run:v2 phase';
      END IF;
    END $$;
    """


def upgrade() -> None:
    if op.get_bind().dialect.name == 'postgresql':
        op.execute(text(_guard(
            pending_requires_null=True,
            admission_requires_abandoned=False,
        )))


def downgrade() -> None:
    if op.get_bind().dialect.name == 'postgresql':
        op.execute(text(_guard(
            pending_requires_null=False,
            admission_requires_abandoned=True,
        )))
