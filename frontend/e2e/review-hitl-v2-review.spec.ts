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

test("same-route workflow navigation replaces the queue before fetching the new context", async ({ page }) => {
  const secondThreadId = "workflow-thread-2";
  const secondItem = { ...item, id: 2002, payload: { ...item.payload, title: "Second workflow candidate" } };
  const requestedThreads: string[] = [];
  await installReviewRoutes(page);
  await page.route("**/api/v1/review?status=pending_review**", async (route) => {
    const workflowThreadId = new URL(route.request().url()).searchParams.get("workflow_thread_id");
    requestedThreads.push(workflowThreadId ?? "");
    const items = workflowThreadId === secondThreadId ? [secondItem] : [item];
    await route.fulfill({ contentType: "application/json", json: {
      groups: [{ group_id: `history_event:${workflowThreadId}`, title: items[0].payload.title, item_type: "history_event", status: "pending_review", permission_level: "internal", items, total_count: 1, avg_confidence: 0.9 }],
      items,
      total_count: 1,
      limit: 50,
      offset: 0,
      has_more: false,
      include_previews: false,
    } });
  });
  await page.route(`**/api/v1/orchestration/v2/company-memory/runs/${secondThreadId}`, (route) => route.fulfill({ contentType: "application/json", json: workflowStatus({ thread_id: secondThreadId, review_item_count: 1, review_status_counts: { pending_review: 1 } }) }));

  await page.goto(`/review?workflow_thread_id=${threadId}&itemId=${item.id}`);
  await expect(page.getByText("Workflow candidate", { exact: true })).toBeVisible();
  await page.evaluate((nextUrl) => {
    window.history.pushState({}, "", nextUrl);
    window.dispatchEvent(new PopStateEvent("popstate"));
  }, `/review?workflow_thread_id=${secondThreadId}&itemId=${secondItem.id}`);

  await expect(page.getByText("Second workflow candidate", { exact: true })).toBeVisible();
  await expect(page.getByText("Workflow candidate", { exact: true })).toHaveCount(0);
  expect(requestedThreads).toEqual([threadId, secondThreadId]);
});

test("late responses from a superseded workflow cannot overwrite the newer queue", async ({ page }) => {
  const secondThreadId = "workflow-thread-2";
  let releaseFirst: (() => void) | undefined;
  const firstResponse = new Promise<void>((resolve) => { releaseFirst = resolve; });
  await installReviewRoutes(page);
  await page.route("**/api/v1/review?status=pending_review**", async (route) => {
    const workflowThreadId = new URL(route.request().url()).searchParams.get("workflow_thread_id");
    if (workflowThreadId === threadId) await firstResponse;
    const currentItem = workflowThreadId === secondThreadId
      ? { ...item, id: 2002, payload: { ...item.payload, title: "New authoritative candidate" } }
      : item;
    await route.fulfill({ contentType: "application/json", json: {
      groups: [{ group_id: `history_event:${workflowThreadId}`, title: currentItem.payload.title, item_type: "history_event", status: "pending_review", permission_level: "internal", items: [currentItem], total_count: 1, avg_confidence: 0.9 }],
      items: [currentItem], total_count: 1, limit: 50, offset: 0, has_more: false, include_previews: false,
    } });
  });
  await page.route(`**/api/v1/orchestration/v2/company-memory/runs/${secondThreadId}`, (route) => route.fulfill({ contentType: "application/json", json: workflowStatus({ thread_id: secondThreadId }) }));

  await page.goto(`/review?workflow_thread_id=${threadId}`);
  await page.evaluate((nextUrl) => {
    window.history.pushState({}, "", nextUrl);
    window.dispatchEvent(new PopStateEvent("popstate"));
  }, `/review?workflow_thread_id=${secondThreadId}`);
  await expect(page.getByText("New authoritative candidate", { exact: true })).toBeVisible();
  releaseFirst?.();
  await expect(page.getByText("New authoritative candidate", { exact: true })).toBeVisible();
  await expect(page.getByText("Workflow candidate", { exact: true })).toHaveCount(0);
});

test("a lost review mutation response still reconciles the filtered queue and workflow status", async ({ page }) => {
  let listRequests = 0;
  let statusRequests = 0;
  await installReviewRoutes(page);
  await page.route("**/api/v1/review?status=pending_review**", async (route) => {
    listRequests += 1;
    await route.fulfill({ contentType: "application/json", json: {
      groups: [{ group_id: "history_event:workflow", title: "Workflow", item_type: "history_event", status: "pending_review", permission_level: "internal", items: [item], total_count: 1, avg_confidence: 0.9 }],
      items: [item], total_count: 1, limit: 50, offset: 0, has_more: false, include_previews: false,
    } });
  });
  await page.route(`**/api/v1/orchestration/v2/company-memory/runs/${threadId}`, async (route) => {
    statusRequests += 1;
    await route.fulfill({ contentType: "application/json", json: workflowStatus() });
  });
  await page.route(`**/api/v1/review/${item.id}/approve`, (route) => route.fulfill({ status: 503, contentType: "application/json", json: { detail: "response lost" } }));

  await page.goto(`/review?workflow_thread_id=${threadId}`);
  await expect.poll(() => listRequests).toBe(1);
  await expect.poll(() => statusRequests).toBe(1);
  await page.locator(".group-container > div:first-child").click();
  await page.getByTestId(`review-item-${item.id}`).getByRole("button", { name: "승인" }).click();

  await expect.poll(() => listRequests).toBeGreaterThan(1);
  await expect.poll(() => statusRequests).toBeGreaterThan(1);
});

test("checkpoint retry resumes exactly once through the explicit lifecycle action", async ({ page }) => {
  let resumes = 0;
  await installReviewRoutes(page, { status: { status: "checkpoint_failed", retry_allowed: true }, onResume: () => { resumes += 1; } });
  await page.goto(`/review?workflow_thread_id=${threadId}`);

  await page.getByRole("button", { name: "다시 시도" }).click();
  await expect.poll(() => resumes).toBe(1);
});

test("lifecycle states use Korean labels and bounded recovery context", async ({ page }) => {
  await installReviewRoutes(page, { status: { status: "drafting" } });
  await page.goto(`/review?workflow_thread_id=${threadId}`);
  await expect(page.getByTestId("review-workflow-context")).toContainText("상태: 후보 작성 중");
  await expect(page.getByTestId("review-workflow-context")).toContainText("연결된 검토 항목만 표시합니다");
  await expect(page.getByTestId("review-workflow-context")).toContainText("워크플로 버전 company-memory-review-v2.0");

  await installReviewRoutes(page, { status: { status: "failed" } });
  await page.goto(`/review?workflow_thread_id=${threadId}`);
  await expect(page.getByTestId("review-workflow-context")).toContainText("Integrations에서 다시 동기화");
});

test("filtered pagination preserves the workflow filter and item deep link behavior", async ({ page }) => {
  const urls: string[] = [];
  const nextItem = { ...item, id: 1002, payload: { ...item.payload, title: "Second filtered candidate" } };
  await installReviewRoutes(page);
  await page.route("**/api/v1/review?status=pending_review**", async (route) => {
    const url = route.request().url();
    urls.push(url);
    const offset = Number(new URL(url).searchParams.get("offset"));
    const items = offset === 0 ? [item] : [nextItem];
    await route.fulfill({ contentType: "application/json", json: {
      groups: [{ group_id: "history_event:workflow", title: "Workflow", item_type: "history_event", status: "pending_review", permission_level: "internal", items, total_count: 2, avg_confidence: 0.9 }],
      items, total_count: 2, limit: 50, offset, has_more: offset === 0, include_previews: false,
    } });
  });

  await page.goto(`/review?workflow_thread_id=${threadId}&itemId=${item.id}`);
  await expect(page.getByTestId(`review-item-${item.id}`)).toBeVisible();
  await page.getByRole("button", { name: "더 보기" }).click();
  await expect(page.getByTestId(`review-item-${nextItem.id}`)).toBeVisible();
  expect(urls.map((url) => new URL(url).searchParams.get("workflow_thread_id"))).toEqual([threadId, threadId]);
  expect(urls.map((url) => new URL(url).searchParams.get("offset"))).toEqual(["0", "50"]);
});

test("in-progress lifecycle states have distinct Korean guidance", async ({ page }) => {
  for (const [status, label, guidance] of [
    ["created", "시작 준비", "시작할 준비"],
    ["drafting", "후보 작성 중", "검토 후보를 작성"],
    ["checkpoint_pending", "검토 대기 저장 중", "안전하게 저장"],
    ["resuming", "검토 결과 반영 중", "검토 결과를 반영"],
  ] as const) {
    await installReviewRoutes(page, { status: { status } });
    await page.goto(`/review?workflow_thread_id=${threadId}`);
    await expect(page.getByTestId("review-workflow-context")).toContainText(`상태: ${label}`);
    await expect(page.getByTestId("review-workflow-context")).toContainText(guidance);
  }
});
