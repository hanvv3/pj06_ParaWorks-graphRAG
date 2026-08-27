import type {
  ReviewWorkflowDiagnostic,
  ReviewWorkflowDryRun,
  ReviewWorkflowLifecycleStatus,
} from "@/lib/api/types";

type LaunchState =
  | "loading_preview"
  | "ready"
  | "launching"
  | "error"
  | "no_candidates"
  | "terminal";

type ReviewCandidateLaunchPanelProps = {
  diagnostic: ReviewWorkflowDiagnostic;
  dryRun?: ReviewWorkflowDryRun;
  launchState: LaunchState;
  terminalStatus?: ReviewWorkflowLifecycleStatus;
  errorMessage?: string;
  onLaunch: () => void;
  onRetryPreview: () => void;
};

function formatUsd(value: number) {
  return `$${value.toFixed(4)}`;
}

/** Renders a bounded V2 cost preview and leaves all API calls to its parent. */
export function ReviewCandidateLaunchPanel({
  diagnostic,
  dryRun,
  launchState,
  terminalStatus,
  errorMessage,
  onLaunch,
  onRetryPreview,
}: ReviewCandidateLaunchPanelProps) {
  const previewLoading = launchState === "loading_preview";
  const launching = launchState === "launching";
  const budgetExceeded = dryRun?.budget_status === "over_budget";
  const terminal = launchState === "terminal" || launchState === "no_candidates";
  const disabled = previewLoading || launching || budgetExceeded || terminal || !dryRun;

  return (
    <section
      data-testid="review-candidate-launch-panel"
      className="mt-4 rounded-lg border border-[var(--line-soft)] bg-[var(--glass-strong)] p-4"
      aria-live="polite"
    >
      <div>
        <h3 className="text-sm font-semibold text-[var(--ink-strong)]">검토 후보 미리보기</h3>
        <p className="mt-1 text-xs leading-5 text-[var(--ink-muted)]">
          이번 동기화에서 확인된 변경 근거만 사용합니다.
        </p>
      </div>

      {previewLoading ? <p className="mt-3 text-sm text-[var(--ink-muted)]">비용을 확인하고 있습니다.</p> : null}

      {dryRun ? (
        <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
          <Metric label="변경 원본" value={`${dryRun.source_count.toLocaleString()}개`} />
          <Metric label="예상 비용" value={formatUsd(dryRun.estimated_cost_usd)} />
          <Metric label="예상 입력 토큰" value={dryRun.estimated_input_tokens.toLocaleString()} />
          <Metric label="예상 출력 토큰" value={dryRun.estimated_output_tokens.toLocaleString()} />
        </div>
      ) : null}

      {dryRun?.budget_status === "within_budget" ? (
        <p className="mt-3 text-sm font-medium text-emerald-700">예산 이내</p>
      ) : null}

      {dryRun?.budget_status === "over_budget" ? (
        <p className="mt-3 text-sm font-medium text-rose-700">예산 초과로 검토 후보를 만들 수 없습니다.</p>
      ) : null}
      {dryRun?.cache_hit || dryRun?.budget_status === "cached" ? (
        <p className="mt-3 text-sm font-medium text-emerald-700">캐시 재사용 · 추가 모델 호출 없음</p>
      ) : null}
      {dryRun?.budget_status === "no_input" ? (
        <p className="mt-3 text-sm font-medium text-emerald-700">추가 비용 없음</p>
      ) : null}
      {!diagnostic.durable ? (
        <p className="mt-3 text-xs leading-5 text-[var(--ink-muted)]">
          메모리 모드에서는 서버 재시작 뒤 검토 흐름을 이어갈 수 없습니다.
        </p>
      ) : null}
      {launchState === "no_candidates" ? (
        <p className="mt-3 text-sm font-medium text-[var(--ink-strong)]">새 검토 후보 없음</p>
      ) : null}
      {launchState === "terminal" ? (
        <p className="mt-3 text-sm font-medium text-[var(--ink-strong)]">
          {terminalStatus === "cancelled"
            ? "작업 취소됨 — 이 동기화 배치의 검토 후보 만들기는 다시 실행할 수 없습니다."
            : terminalStatus === "failed"
              ? "작업 실패 — 이 동기화 배치의 검토 후보 만들기는 다시 실행할 수 없습니다."
              : "검토 후보 만들기가 완료되지 않았습니다. 잠시 후 새 동기화를 시작해 주세요."}
        </p>
      ) : null}
      {launchState === "error" ? (
        <p className="mt-3 text-sm font-medium text-rose-700">
          {errorMessage ?? "요청을 처리할 수 없습니다. 잠시 후 다시 시도해 주세요."} 다시 시도할 수 있습니다.
        </p>
      ) : null}

      {launchState === "error" && !dryRun ? (
        <button
          type="button"
          onClick={onRetryPreview}
          className="liquid-control mt-3 inline-flex h-9 w-full items-center justify-center rounded-lg px-3 text-sm font-semibold"
        >
          미리보기 다시 시도
        </button>
      ) : null}

      <button
        type="button"
        onClick={onLaunch}
        disabled={disabled}
        className="liquid-primary mt-4 inline-flex h-9 w-full items-center justify-center rounded-lg px-3 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-55"
      >
        {launching ? "검토 후보를 만들고 있습니다" : "검토 후보 만들기"}
      </button>
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-[var(--glass-elevated)] px-3 py-2">
      <p className="text-xs text-[var(--ink-muted)]">{label}</p>
      <p className="mt-1 font-semibold text-[var(--ink-strong)]">{value}</p>
    </div>
  );
}
