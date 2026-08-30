"use client";

import { AutoReviewActions } from "@/components/review/AutoReviewActions";
import { AutoReviewBadge } from "@/components/review/AutoReviewBadge";
import type { ReviewItem } from "@/lib/api/types";

export function AutoReviewTrustPanel({ item, onChanged }: { item: ReviewItem; onChanged: () => Promise<void> }) {
  const summary = item.auto_review_summary; const audit = item.auto_review_audit;
  return <section className="mt-4 rounded-lg border border-blue-200 bg-blue-50/70 p-4" data-testid="auto-review-trust-panel">
    <div className="flex flex-wrap items-center gap-2"><AutoReviewBadge resolutionSource="auto_policy" />{audit?.status === "pending" ? <span className="badge warning">감사 필요</span> : null}{audit?.action_required || audit?.status === "remediation_required" ? <span className="badge warning">조치 필요</span> : null}</div>
    {summary ? <dl className="mt-3 grid gap-2 text-xs sm:grid-cols-2"><div><dt className="text-muted">검증 모델</dt><dd className="font-semibold">Terra · medium</dd></div><div><dt className="text-muted">정책</dt><dd className="font-semibold">자동 검토 정책 v1</dd></div><div><dt className="text-muted">근거 일치</dt><dd className="font-semibold">{summary.supported_substantive_field_count}개 · 최소 {(summary.minimum_entailment_score * 100).toFixed(0)}%</dd></div><div><dt className="text-muted">검증 시각</dt><dd className="font-semibold">{new Date(summary.validated_at).toLocaleString("ko-KR")}</dd></div></dl> : <p className="mt-3 text-sm text-muted">검증 요약을 표시할 수 없습니다.</p>}
    <AutoReviewActions item={item} onChanged={onChanged} />
  </section>;
}
