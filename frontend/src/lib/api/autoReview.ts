import { ApiError, apiPost } from "./client";
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

const AUTO_REVIEW_ERROR_CODES = {
  remediation_required: true,
} as const;

function isAutoReviewErrorCode(value: unknown): value is keyof typeof AUTO_REVIEW_ERROR_CODES {
  return typeof value === "string"
    && Object.prototype.hasOwnProperty.call(AUTO_REVIEW_ERROR_CODES, value);
}

export function isAutoReviewRemediationError(error: unknown) {
  if (error instanceof ApiError) return isAutoReviewErrorCode(error.code);
  if (!(error instanceof Error) || error.message.length > 256) return false;
  try {
    const parsed = JSON.parse(error.message) as unknown;
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return false;
    const keys = Reflect.ownKeys(parsed);
    if (keys.length !== 1 || keys[0] !== "code") return false;
    const descriptor = Object.getOwnPropertyDescriptor(parsed, "code");
    return descriptor !== undefined
      && Object.prototype.hasOwnProperty.call(descriptor, "value")
      && isAutoReviewErrorCode(descriptor.value);
  } catch {
    return false;
  }
}
