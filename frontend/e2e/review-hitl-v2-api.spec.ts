import { expect, test } from "@playwright/test";

import {
  cancelReviewWorkflow,
  dryRunReviewWorkflow,
  getReviewWorkflowDiagnostic,
  getReviewWorkflowStatus,
  launchReviewWorkflow,
  resumeReviewWorkflow,
} from "../src/lib/api/reviewWorkflow";

const request = {
  source_refs: [
    {
      source_type: "gmail" as const,
      source_id: "gmail:message-1",
      version_or_signature: "signature-1",
    },
  ],
  agent_names: ["mail_document_agent"],
  client_request_id: "client-request-1",
};

const status = {
  thread_id: "thread id/with ? punctuation",
  status: "awaiting_human_review" as const,
  review_item_count: 1,
  review_status_counts: { pending_review: 1 },
  durable: false,
  graph_version: "company-memory-review-v2.0",
  review_resolution_ready: false,
  checkpoint_resumable: true,
  resume_allowed: false,
  retry_allowed: false,
  created_at: "2026-08-27T00:00:00Z",
  updated_at: "2026-08-27T00:00:00Z",
  error_code: null,
  resume_error_code: null,
};

test("review workflow wrappers use the approved methods, paths, and launch payload", async () => {
  const originalFetch = globalThis.fetch;
  const calls: Array<{ path: string; method: string; body?: unknown }> = [];

  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({
      path: new URL(String(input)).pathname,
      method: init?.method ?? "GET",
      body: typeof init?.body === "string" ? JSON.parse(init.body) : undefined,
    });
    return new Response(JSON.stringify(status), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  try {
    await getReviewWorkflowDiagnostic();
    await dryRunReviewWorkflow(request);
    await launchReviewWorkflow(request);
    await getReviewWorkflowStatus(status.thread_id);
    await resumeReviewWorkflow(status.thread_id);
    await cancelReviewWorkflow(status.thread_id);
  } finally {
    globalThis.fetch = originalFetch;
  }

  expect(calls).toEqual([
    { path: "/api/v1/orchestration/v2/company-memory", method: "GET" },
    {
      path: "/api/v1/orchestration/v2/company-memory/dry-run",
      method: "POST",
      body: request,
    },
    {
      path: "/api/v1/orchestration/v2/company-memory/runs",
      method: "POST",
      body: request,
    },
    {
      path: "/api/v1/orchestration/v2/company-memory/runs/thread%20id%2Fwith%20%3F%20punctuation",
      method: "GET",
    },
    {
      path: "/api/v1/orchestration/v2/company-memory/runs/thread%20id%2Fwith%20%3F%20punctuation/resume",
      method: "POST",
    },
    {
      path: "/api/v1/orchestration/v2/company-memory/runs/thread%20id%2Fwith%20%3F%20punctuation/cancel",
      method: "POST",
    },
  ]);
});

test("review workflow wrappers expose only a bounded backend error code", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => new Response(
    JSON.stringify({ detail: { code: "evidence_changed" } }),
    { status: 409, headers: { "Content-Type": "application/json" } },
  )) as typeof fetch;

  try {
    await expect(launchReviewWorkflow(request)).rejects.toMatchObject({
      code: "evidence_changed",
      message: "요청을 처리할 수 없습니다. 잠시 후 다시 시도해 주세요.",
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});
