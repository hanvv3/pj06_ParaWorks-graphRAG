"use client";

import { useState } from "react";
import { isAutoReviewRemediationError, revokeAutoApproval, submitAutoReviewAudit, type AutoReviewRevokeReasonCode } from "@/lib/api/autoReview";
import type { AutoReviewAuditOutcome, ReviewItem } from "@/lib/api/types";

const REVOKE_REASONS: Array<{ value: AutoReviewRevokeReasonCode; label: string }> = [
  { value: "business_withdrawal", label: "업무상 철회" }, { value: "incorrect_content", label: "내용 오류" },
  { value: "permission_violation", label: "권한 위반" }, { value: "wrong_source_version", label: "원본 버전 오류" },
  { value: "policy_violation", label: "정책 위반" },
];

export function AutoReviewActions({ item, onChanged }: { item: ReviewItem; onChanged: () => Promise<void> }) {
  const [revokeReason, setRevokeReason] = useState<AutoReviewRevokeReasonCode>("business_withdrawal");
  const [auditOutcome, setAuditOutcome] = useState<AutoReviewAuditOutcome>("confirmed");
  const [auditReason, setAuditReason] = useState("");
  const [pending, setPending] = useState(false);
  const [message, setMessage] = useState<string>();

  async function run(action: () => Promise<unknown>, success: string) {
    if (pending) return;
    setPending(true); setMessage(undefined);
    try { await action(); setMessage(success); }
    catch (error) { setMessage(isAutoReviewRemediationError(error) ? "조치 필요" : "요청을 처리하지 못했습니다. 현재 상태를 다시 확인해 주세요."); }
    finally { await onChanged(); setPending(false); }
  }

  return <div className="mt-4 space-y-3 border-t border-dashed border-[var(--line-soft)] pt-4">
    <div className="flex flex-wrap items-end gap-2">
      <label className="text-xs font-semibold">철회 사유<select aria-label="자동 승인 철회 사유" value={revokeReason} onChange={(event) => setRevokeReason(event.target.value as AutoReviewRevokeReasonCode)} className="mt-1 block h-9 rounded-lg border px-3 text-sm">{REVOKE_REASONS.map((reason) => <option key={reason.value} value={reason.value}>{reason.label}</option>)}</select></label>
      <button type="button" disabled={pending} onClick={() => void run(() => revokeAutoApproval(item.id, revokeReason), "철회 결과를 반영했습니다.")} className="h-9 rounded-lg border px-3 text-sm font-semibold disabled:opacity-50">자동 승인 철회</button>
    </div>
    <div className="grid gap-2 sm:grid-cols-[180px_1fr_auto] sm:items-end">
      <label className="text-xs font-semibold">감사 결과<select aria-label="자동 승인 감사 결과" value={auditOutcome} onChange={(event) => setAuditOutcome(event.target.value as AutoReviewAuditOutcome)} className="mt-1 block h-9 w-full rounded-lg border px-3 text-sm"><option value="confirmed">근거 확인</option><option value="incorrect">내용 오류</option><option value="permission_violation">권한 위반</option><option value="source_version_violation">원본 버전 오류</option><option value="policy_violation">정책 위반</option></select></label>
      <label className="text-xs font-semibold">검토 사유<input aria-label="자동 승인 감사 사유" value={auditReason} onChange={(event) => setAuditReason(event.target.value)} maxLength={500} className="mt-1 block h-9 w-full rounded-lg border px-3 text-sm" /></label>
      <button type="button" disabled={pending || auditReason.trim().length === 0} onClick={() => void run(() => submitAutoReviewAudit(item.id, auditOutcome, auditReason), "감사 결과를 반영했습니다.")} className="h-9 rounded-lg bg-[#21132b] px-3 text-sm font-semibold text-white disabled:opacity-50">감사 반영</button>
    </div>
    {message ? <p className="text-sm font-semibold text-amber-800" role="status">{message}</p> : null}
  </div>;
}
