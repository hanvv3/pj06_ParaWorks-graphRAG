import { expect, test } from "@playwright/test";
import { revokeAutoApproval, submitAutoReviewAudit } from "../src/lib/api/autoReview";

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
