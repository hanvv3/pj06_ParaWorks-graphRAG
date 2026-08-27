import { expect, test, type Page } from "@playwright/test";

const threadId = "workflow-thread-1";

const item = {
  id: 1001,
  item_type: "history_event",
  payload: { title: "Workflow candidate", summary: "A reviewed workflow candidate.", agent_name: "mail_document_agent" },
  source_links: ["https://mail.mock/messages/1001"],
  source_snippets: ["Workflow evidence."],
  source_evidence: [],
  agent_run_id: 1,
  agent_run_details: { model_name: "gpt-5-mini", prompt_version: "mail:v1", estimated_cost_usd: 0.0001, total_tokens: 10 },
  confidence_score: 0.9,
  permission_level: "internal",
  status: "pending_review",
  reviewer_id: null,
};

function workflowStatus(overrides: Record<string, unknown> = {}) {
  return {
    thread_id: threadId,
    status: "awaiting_human_review",
    review_item_count: 3,
    review_status_counts: { pending_review: 1, approved: 2 },
    durable: true,
    graph_version: "company-memory-review-v2.0",
    review_resolution_ready: false,
    checkpoint_resumable: true,
    resume_allowed: false,
    retry_allowed: false,
    created_at: "2026-08-27T00:00:00Z",
    updated_at: "2026-08-27T00:00:00Z",
    error_code: null,
    resume_error_code: null,
    ...overrides,
  };
}

async function installReviewRoutes(
  page: Page,
  options: {
    status?: Record<string, unknown>;
    items?: typeof item[];
    onReviewRequest?: (url: string) => void;
    statusFailure?: { detail: string };
    onResume?: () => void;
  } = {},
) {
  const items = options.items ?? [item];
  await page.addInitScript(() => window.localStorage.setItem("paraworks-demo-user", "demo-admin"));
  await page.route("**/api/v1/auth/me", (route) => route.fulfill({ contentType: "application/json", json: { user: { id: "demo-admin", email: "admin@paraworks.com", role: "admin", permission_levels: ["public", "internal", "restricted"], name: "Admin", title: "Admin", department: "Platform" } } }));
  await page.route("**/api/v1/notifications", (route) => route.fulfill({ contentType: "application/json", json: { counts: { total: 0 }, notifications: [] } }));
  await page.route("**/api/v1/dashboard", (route) => route.fulfill({ contentType: "application/json", json: { pending_review_count: items.length, source_counts: {} } }));
  await page.route("**/api/v1/projects/defined", (route) => route.fulfill({ contentType: "application/json", json: { projects: [] } }));
  await page.route("**/api/v1/review?status=pending_review**", async (route) => {
    options.onReviewRequest?.(route.request().url());
    await route.fulfill({
      contentType: "application/json",
      json: {
        groups: items.length ? [{ group_id: "history_event:workflow", title: "Workflow", item_type: "history_event", status: "pending_review", permission_level: "internal", items, total_count: items.length, avg_confidence: 0.9 }] : [],
        items,
        total_count: items.length,
        limit: 50,
        offset: 0,
        has_more: false,
        include_previews: false,
      },
    });
  });
  await page.route(`**/api/v1/orchestration/v2/company-memory/runs/${threadId}/resume`, async (route) => {
    options.onResume?.();
    await route.fulfill({ contentType: "application/json", json: workflowStatus({ status: "completed" }) });
  });
  await page.route(`**/api/v1/orchestration/v2/company-memory/runs/${threadId}`, async (route) => {
    if (options.statusFailure) {
      await route.fulfill({ status: 404, contentType: "application/json", json: options.statusFailure });
      return;
    }
    await route.fulfill({ contentType: "application/json", json: workflowStatus(options.status) });
  });
  for (const action of ["approve", "reject", "request-more-evidence"]) {
    await page.route(`**/api/v1/review/${item.id}/${action}`, (route) => route.fulfill({ contentType: "application/json", json: { ...item, status: action === "approve" ? "approved" : action === "reject" ? "rejected" : "needs_more_evidence" } }));
  }
}

test("filters review items and enables explicit completion only when ready", async ({ page }) => {
  let resumes = 0;
  await installReviewRoutes(page, { status: { review_resolution_ready: true, resume_allowed: true }, onResume: () => { resumes += 1; } });
  await page.goto(`/review?workflow_thread_id=${threadId}`);

  await expect(page.getByTestId("review-workflow-context")).toContainText("2 / 3");
  await expect(page.getByRole("button", { name: "검토 완료" })).toBeEnabled();
  await page.getByRole("button", { name: "검토 완료" }).click();
  await expect.poll(() => resumes).toBe(1);
});

test("review item actions never auto resume the workflow", async ({ page }) => {
  let resumes = 0;
  await installReviewRoutes(page, { onResume: () => { resumes += 1; } });
  await page.goto(`/review?workflow_thread_id=${threadId}`);
  await page.locator(".group-container > div:first-child").click();
  const reviewItem = page.getByTestId(`review-item-${item.id}`);
  await reviewItem.getByRole("button", { name: "승인" }).click();
  await reviewItem.getByRole("button", { name: "반려" }).click();
  await reviewItem.getByRole("button", { name: "근거 추가 요청" }).click();
  await page.getByRole("button", { name: "요청 보내기" }).click();

  expect(resumes).toBe(0);
});

test("needs more evidence ends with guidance and no retry", async ({ page }) => {
  await installReviewRoutes(page, { status: { status: "needs_more_evidence" } });
  await page.goto(`/review?workflow_thread_id=${threadId}`);

  await expect(page.getByTestId("review-workflow-context")).toContainText("추가 근거");
  await expect(page.getByTestId("review-workflow-context")).not.toContainText("다시 시도");
  await expect(page.getByTestId("review-workflow-context")).not.toContainText("검토 완료");
});

test("checkpoint failed is the only retryable state", async ({ page }) => {
  await installReviewRoutes(page, { status: { status: "checkpoint_failed", retry_allowed: true } });
  await page.goto(`/review?workflow_thread_id=${threadId}`);

  await expect(page.getByRole("button", { name: "다시 시도" })).toBeVisible();
  await expect(page.getByTestId("review-workflow-context")).not.toContainText("검토 완료");
});

test("failed and cancelled states show safe terminal guidance", async ({ page }) => {
  await installReviewRoutes(page, { status: { status: "failed" } });
  await page.goto(`/review?workflow_thread_id=${threadId}`);
  await expect(page.getByTestId("review-workflow-context")).toContainText("작업을 완료하지 못했습니다");
  await expect(page.getByTestId("review-workflow-context")).not.toContainText("다시 시도");

  await installReviewRoutes(page, { status: { status: "cancelled" } });
  await page.goto(`/review?workflow_thread_id=${threadId}`);
  await expect(page.getByTestId("review-workflow-context")).toContainText("작업이 취소되었습니다");
});

test("memory mode explains process local durability", async ({ page }) => {
  await installReviewRoutes(page, { status: { durable: false } });
  await page.goto(`/review?workflow_thread_id=${threadId}`);

  await expect(page.getByTestId("review-workflow-context")).toContainText("현재 프로세스에서만 유지");
});

test("workflow panel stacks above queue actions on mobile", async ({ page }) => {
  await installReviewRoutes(page);
  await page.goto(`/review?workflow_thread_id=${threadId}`);

  const panel = await page.getByTestId("review-workflow-context").boundingBox();
  const queueActions = await page.getByTestId("review-approve-loaded").boundingBox();
  expect(panel?.y).toBeLessThan(queueActions?.y ?? 0);
});

test("the first review request already contains the workflow filter", async ({ page }) => {
  const urls: string[] = [];
  await installReviewRoutes(page, { onReviewRequest: (url) => urls.push(url) });
  await page.goto(`/review?workflow_thread_id=${threadId}&itemId=${item.id}`);

  await expect.poll(() => urls.length).toBeGreaterThan(0);
  expect(new URL(urls[0]).searchParams.get("workflow_thread_id")).toBe(threadId);
});

test("status 404 hides the thread id and raw detail behind generic copy", async ({ page }) => {
  const rawDetail = "provider detail must remain hidden";
  await installReviewRoutes(page, { statusFailure: { detail: rawDetail } });
  await page.goto(`/review?workflow_thread_id=${threadId}`);

  await expect(page.getByTestId("review-workflow-context")).toContainText("워크플로 정보를 불러올 수 없습니다");
  await expect(page.locator("body")).not.toContainText(threadId);
  await expect(page.locator("body")).not.toContainText(rawDetail);
});

test("an empty hidden workflow keeps group and bulk controls safe", async ({ page }) => {
  await installReviewRoutes(page, { items: [] });
  await page.goto(`/review?workflow_thread_id=${threadId}`);

  await expect(page.locator(".group-container")).toHaveCount(0);
  await expect(page.getByTestId("review-approve-loaded")).toBeDisabled();
  await expect(page.getByTestId("review-bulk-approve")).toBeDisabled();
});
