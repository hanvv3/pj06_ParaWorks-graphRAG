import { apiPost } from "./client";
import type { AutoReviewAuditOutcome } from "./types";

export type AutoReviewRevokeReasonCode = "business_withdrawal" | "incorrect_content" | "permission_violation" | "wrong_source_version" | "policy_violation";

export type AutoReviewRevokeResponse = {
  review_item_id: number;
  status: "revoked";
  replayed: boolean;
  knowledge_remains_trusted: boolean;
  revoked_document_count: number;
};

export type AutoReviewAuditResponse = {
  audit_status: "completed" | "remediation_required";
  breaker_open: boolean;
  revoke_status: "not_required" | "revoked" | "remediation_required";
};

export function revokeAutoApproval(reviewItemId: number, reasonCode: AutoReviewRevokeReasonCode) {
  return apiPost<AutoReviewRevokeResponse>(`/api/v1/review/${reviewItemId}/revoke-auto-approval`, { reason_code: reasonCode });
}

export function submitAutoReviewAudit(reviewItemId: number, outcome: AutoReviewAuditOutcome, reason: string) {
  return apiPost<AutoReviewAuditResponse>(`/api/v1/review/${reviewItemId}/auto-review-audit`, { outcome, reason });
}

export function isAutoReviewRemediationError(error: unknown) {
  if (!(error instanceof Error) || error.message.length > 256) return false;
  try {
    const parsed = JSON.parse(error.message) as { code?: unknown };
    return Object.keys(parsed).length === 1 && parsed.code === "remediation_required";
  } catch {
    return false;
  }
}
