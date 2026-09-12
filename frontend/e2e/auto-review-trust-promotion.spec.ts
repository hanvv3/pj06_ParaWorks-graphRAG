import { expect, test } from "@playwright/test";
import { isAutoReviewRemediationError, revokeAutoApproval, submitAutoReviewAudit } from "../src/lib/api/autoReview";
import { ApiError } from "../src/lib/api/client";

async function captureAutoReviewError(body: string) {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => new Response(body, {
    status: 409,
    headers: { "Content-Type": "application/json" },
  })) as typeof fetch;
  try {
    await submitAutoReviewAudit(9, "incorrect", "원문과 불일치");
    throw new Error("expected submitAutoReviewAudit to reject");
  } catch (error) {
    if (!(error instanceof ApiError)) throw error;
    return error;
  } finally {
    globalThis.fetch = originalFetch;
  }
}

test("auto review wrappers send only the strict inline action bodies", async () => {
  const originalFetch = globalThis.fetch;
  const bodies: unknown[] = [];
  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    bodies.push(typeof init?.body === "string" ? JSON.parse(init.body) : undefined);
    return new Response(JSON.stringify({ status: "ok" }), { status: 200, headers: { "Content-Type": "application/json" } });
  }) as typeof fetch;
  try {
    await revokeAutoApproval(9, "permission_violation");
    await submitAutoReviewAudit(9, "confirmed", "원문 근거 확인");
  } finally {
    globalThis.fetch = originalFetch;
  }
  expect(bodies).toEqual([
    { reason_code: "permission_violation" },
    { outcome: "confirmed", reason: "원문 근거 확인" },
  ]);
});

test("real Auto-Review transport preserves only remediation_required for local revalidation", async () => {
  const observed = await captureAutoReviewError(JSON.stringify({
    detail: { code: "remediation_required" },
  }));

  expect(observed).toBeInstanceOf(ApiError);
  expect(observed.status).toBe(409);
  expect(observed.code).toBe("remediation_required");
  expect(observed.message).toBe("요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.");
  expect(isAutoReviewRemediationError(observed)).toBe(true);
});

test("Auto-Review rejects unknown, raw, and non-exact remediation failures", async () => {
  for (const body of [
    JSON.stringify({ detail: { code: "private_auto_review_code" } }),
    JSON.stringify({ detail: { code: "remediation_required" }, unexpected: true }),
    "Database rollback failed with private state",
  ]) {
    const observed = await captureAutoReviewError(body);
    expect(observed.status).toBe(409);
    expect(observed.code).toBeNull();
    expect(observed.message).toBe("요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.");
    expect(isAutoReviewRemediationError(observed)).toBe(false);
  }
});

test("Auto-Review legacy serialized error compatibility is exact and bounded", () => {
  expect(isAutoReviewRemediationError(new Error('{"code":"remediation_required"}'))).toBe(true);
  expect(isAutoReviewRemediationError(new Error('{"code":"remediation_required","extra":true}'))).toBe(false);
  expect(isAutoReviewRemediationError(new Error('{"code":"private_auto_review_code"}'))).toBe(false);
  expect(isAutoReviewRemediationError(new Error("raw English failure"))).toBe(false);
  expect(isAutoReviewRemediationError(new Error("x".repeat(257)))).toBe(false);
});
