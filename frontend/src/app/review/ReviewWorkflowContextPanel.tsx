"use client";

import { CheckCircle2, RefreshCw } from "lucide-react";
import type { ReviewWorkflowStatus } from "@/lib/api/types";

type ReviewWorkflowContextPanelProps = {
  status?: ReviewWorkflowStatus;
  loading: boolean;
  unavailable: boolean;
  actionPending: boolean;
  onResume: () => void;
};

function resolutionCount(status: ReviewWorkflowStatus) {
  const pending = status.review_status_counts.pending_review ?? 0;
  return Math.max(0, status.review_item_count - pending);
}

function lifecycleLabel(status: ReviewWorkflowStatus) {
  const labels: Record<ReviewWorkflowStatus["status"], string> = {
    created: "시작 준비",
    drafting: "후보 작성 중",
    checkpoint_pending: "검토 대기 저장 중",
    awaiting_human_review: "사람 검토 대기",
    resuming: "검토 결과 반영 중",
    completed: "완료",
    needs_more_evidence: "추가 근거 필요",
    checkpoint_failed: "체크포인트 저장 실패",
    failed: "작업 실패",
    cancelled: "작업 취소",
  };
  return labels[status.status];
}

function guidance(status: ReviewWorkflowStatus) {
  switch (status.status) {
    case "created":
      return "워크플로를 시작할 준비를 하고 있습니다. 잠시 후 검토 후보를 확인하세요.";
    case "drafting":
      return "연결된 근거에서 검토 후보를 작성하고 있습니다. 완료될 때까지 검토 완료를 선택할 수 없습니다.";
    case "checkpoint_pending":
      return "검토 대기 상태를 안전하게 저장하고 있습니다. 저장이 끝나면 검토 항목을 확인하세요.";
    case "awaiting_human_review":
      return "검토 항목을 모두 처리한 뒤 검토 완료를 선택하세요.";
    case "resuming":
      return "명시적으로 완료한 검토 결과를 반영하고 있습니다. 이 단계에서는 추가 동작이 필요하지 않습니다.";
    case "checkpoint_failed":
      return "체크포인트를 저장하지 못했습니다. 다시 시도하기 전에 현재 검토 상태를 확인하세요.";
    case "completed":
      return "검토 워크플로가 완료되었습니다. 승인된 내용은 다음 처리 단계에 반영됩니다.";
    case "needs_more_evidence":
      return "추가 근거가 필요합니다. 원본을 보강하거나 다시 동기화한 뒤 새 검토 후보를 만드세요.";
    case "failed":
      return "작업을 완료하지 못했습니다. 데이터를 변경한 뒤 Integrations에서 다시 동기화하고 새 후보를 만드세요.";
    case "cancelled":
      return "작업이 취소되었습니다. 데이터를 변경한 뒤 Integrations에서 다시 동기화하고 새 후보를 만드세요.";
    default:
      return "워크플로 상태를 확인하고 있습니다.";
  }
}

export function ReviewWorkflowContextPanel({
  status,
  loading,
  unavailable,
  actionPending,
  onResume,
}: ReviewWorkflowContextPanelProps) {
  if (!status && !loading && !unavailable) return null;

  if (unavailable) {
    return (
      <section data-testid="review-workflow-context" className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-950">
        워크플로 정보를 불러올 수 없습니다. 검토 큐는 계속 확인할 수 있습니다.
      </section>
    );
  }

  if (!status) {
    return (
      <section data-testid="review-workflow-context" className="rounded-xl border border-[var(--line-soft)] bg-[var(--glass-elevated)] p-4 text-sm text-[var(--ink-muted)]">
        검토 워크플로 상태를 불러오는 중입니다.
      </section>
    );
  }

  const canResume = status.status === "awaiting_human_review" && status.review_resolution_ready && status.resume_allowed;
  const canRetry = status.status === "checkpoint_failed" && status.retry_allowed;
  const completed = resolutionCount(status);

  return (
    <section data-testid="review-workflow-context" className="rounded-xl border border-[var(--line-soft)] bg-[var(--glass-elevated)] p-4 shadow-sm">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-xs font-bold uppercase tracking-wide text-[var(--ink-muted)]">검토 워크플로</p>
          <p className="mt-1 text-sm font-semibold text-[var(--ink)]">상태: {lifecycleLabel(status)}</p>
          <p className="mt-1 text-base font-bold text-[var(--ink)]">완료 수 / 전체 수: {completed} / {status.review_item_count}</p>
          <p className="mt-1 text-sm leading-6 text-[var(--ink-muted)]">{guidance(status)}</p>
          <p className="mt-1 text-xs text-[var(--ink-muted)]">연결된 검토 항목만 표시합니다 · 워크플로 버전 {status.graph_version}</p>
          <p className="mt-1 text-xs text-[var(--ink-muted)]">
            {status.durable ? "체크포인트가 저장되어 재시작 후에도 이어집니다." : "현재 프로세스에서만 유지됩니다. 서버를 다시 시작하면 이어서 처리할 수 없습니다."}
          </p>
        </div>
        {status.status === "awaiting_human_review" ? (
          <button
            type="button"
            onClick={onResume}
            disabled={!canResume || actionPending}
            className="inline-flex h-10 shrink-0 items-center justify-center gap-2 rounded-lg border border-[#21132b] bg-[#21132b] px-4 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:bg-neutral-400"
          >
            <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
            {actionPending ? "처리 중" : "검토 완료"}
          </button>
        ) : null}
        {canRetry ? (
          <button
            type="button"
            onClick={onResume}
            disabled={actionPending}
            className="inline-flex h-10 shrink-0 items-center justify-center gap-2 rounded-lg border border-[#21132b] bg-[#21132b] px-4 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:bg-neutral-400"
          >
            <RefreshCw className="h-4 w-4" aria-hidden="true" />
            {actionPending ? "처리 중" : "다시 시도"}
          </button>
        ) : null}
      </div>
    </section>
  );
}
