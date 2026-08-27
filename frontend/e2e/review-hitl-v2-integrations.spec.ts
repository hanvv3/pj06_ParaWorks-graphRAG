import { expect, test, type Page } from "@playwright/test";

const sourceRefs = [
  {
    source_type: "gmail",
    source_id: "gmail:message-42",
    version_or_signature: "gmail-signature-42",
  },
  {
    source_type: "drive",
    source_id: "drive:file-9",
    version_or_signature: "drive-revision-9",
  },
] as const;

const diagnostic = {
  enabled: true,
  available: true,
  checkpoint_mode: "memory",
  durable: false,
  graph_version: "company-memory-review-v2.0",
  default_agent_names: ["mail_document_agent", "calendar_agent"],
  error_code: null,
};

const dryRun = {
  workflow_name: "company-memory-review",
  graph_version: "company-memory-review-v2.0",
  source_count: sourceRefs.length,
  agent_names: diagnostic.default_agent_names,
  selection_policy_version: "company-memory-review-selection:v1",
  estimated_input_tokens: 120,
  estimated_output_tokens: 40,
  estimated_cost_usd: 0.0012,
  budget_limit_usd: 0.05,
  budget_status: "within_budget",
  cache_hit: false,
  requires_explicit_run: true,
};

const completedSync = {
  job_id: "gmail-sync-review-v2",
  connector_type: "gmail",
  status: "complete",
  created_review_items: 0,
  pending_review_count: 3,
  fetched_events: sourceRefs.length,
  skipped_events: 0,
  parser_status_counts: {},
  changed_source_ids: ["legacy-id-must-not-be-used"],
  changed_source_refs: sourceRefs,
  agent_generated_items: 0,
  project_assignment_items: 0,
};

async function installIntegrationsRoutes(
  page: Page,
  options: {
    diagnosticResponse?: typeof diagnostic;
    connectorType?: "gmail" | "slack";
    diagnosticFailure?: boolean;
    launchStatuses?: Array<Record<string, unknown>>;
    syncResponses?: Array<Record<string, unknown>>;
    runtimeSyncResponses?: Array<Record<string, unknown> | null>;
    dryRunResponses?: Array<Record<string, unknown>>;
    onDryRun?: (body: unknown) => void;
    onLaunch?: (body: unknown) => void;
  } = {},
) {
  const connectorType = options.connectorType ?? "gmail";
  const diagnosticResponse = options.diagnosticResponse ?? diagnostic;
  const launchStatuses = [...(options.launchStatuses ?? [awaitingReviewStatus()])];
  const syncResponses = [...(options.syncResponses ?? [{ ...completedSync, connector_type: connectorType }])];
  const runtimeSyncResponses = [...(options.runtimeSyncResponses ?? [])];
  const dryRunResponses = [...(options.dryRunResponses ?? [dryRun])];
  let syncRequested = false;
  let latestRuntimeSync: Record<string, unknown> | null = null;

  await page.route("**/api/v1/auth/me", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: {
        user: {
          id: "demo-admin",
          email: "admin@paraworks.com",
          role: "admin",
          permission_levels: ["public", "internal", "restricted"],
          name: "ParaWorks Admin",
          title: "Workspace Administrator",
          department: "Platform",
        },
      },
    });
  });
  await page.route("**/api/v1/integrations", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: [
        {
          type: connectorType,
          display_name: connectorType === "slack" ? "Slack" : "Gmail",
          mode: "live",
          status: "ready",
          auth_type: "oauth",
          required_scopes: [],
          sync_strategy: "incremental",
          cost_policy: "changed sources only",
        },
      ],
    });
  });
  await page.route("**/api/v1/integrations/connections", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: [
        {
          id: "connection-1",
          connector_type: connectorType,
          workspace_name: "ParaWorks",
          status: "connected",
          credential_status: "available",
          scopes: [],
          masked_bot_token: "configured",
        },
      ],
    });
  });
  await page.route("**/api/v1/dashboard", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: {
        pending_review_count: 3,
        source_counts: { slack: 0, gmail: sourceRefs.length, drive: 0, calendar: 0, other: 0 },
      },
    });
  });
  await page.route("**/api/v1/integrations/slack/oauth/install-url", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: { connector_type: "slack", configured: false, install_url: null, state: null, required_scopes: [] },
    });
  });
  await page.route("**/api/v1/integrations/slack/runtime-status", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: {
        connector_type: "slack",
        mode: "live",
        configured_channel_ids: [],
        selected_channel_ids: [],
        channel_options: [],
        connection_status: connectorType === "slack" ? "connected" : "disconnected",
        credential_status: connectorType === "slack" ? "available" : "missing",
        latest_sync: null,
        latest_sync_summary: null,
        last_error: null,
        agent_bridge: { slack_source_count: 0, pending_review_count: 0, ready_for_agent_test: false },
        cost_policy: { status_lookup_triggers_sync: false, status_lookup_triggers_llm: false },
      },
    });
  });
  for (const type of ["gmail", "drive", "calendar"]) {
    await page.route(`**/api/v1/integrations/${type}/oauth/install-url`, async (route) => {
      await route.fulfill({
        contentType: "application/json",
        json: { connector_type: type, configured: false, install_url: null, state: null, required_scopes: [] },
      });
    });
    await page.route(`**/api/v1/integrations/${type}/runtime-status`, async (route) => {
      if (type === connectorType && syncRequested && runtimeSyncResponses.length > 0) {
        latestRuntimeSync = runtimeSyncResponses.shift() ?? latestRuntimeSync;
      }
      await route.fulfill({
        contentType: "application/json",
        json: {
          connector_type: type,
          mode: "live",
          connection_status: type === connectorType ? "connected" : "disconnected",
          credential_status: type === connectorType ? "available" : "missing",
          latest_sync: type === connectorType ? latestRuntimeSync : null,
          cost_policy: { status_lookup_triggers_sync: false, status_lookup_triggers_llm: false },
        },
      });
    });
  }
  await page.route(`**/api/v1/integrations/${connectorType}/sync`, async (route) => {
    syncRequested = true;
    await route.fulfill({
      contentType: "application/json",
      json: syncResponses.shift() ?? { ...completedSync, connector_type: connectorType },
    });
  });
  await page.route("**/api/v1/orchestration/v2/company-memory", async (route) => {
    if (options.diagnosticFailure) {
      await route.fulfill({ status: 500, contentType: "application/json", json: { detail: "diagnostic unavailable" } });
      return;
    }
    await route.fulfill({ contentType: "application/json", json: diagnosticResponse });
  });
  await page.route("**/api/v1/orchestration/v2/company-memory/dry-run", async (route) => {
    options.onDryRun?.(route.request().postDataJSON());
    const next = dryRunResponses.shift() ?? dryRun;
    if (next.status === 500) {
      await route.fulfill({ status: 500, contentType: "application/json", json: { detail: "preview lost" } });
      return;
    }
    await route.fulfill({ contentType: "application/json", json: next });
  });
  await page.route("**/api/v1/orchestration/v2/company-memory/runs", async (route) => {
    options.onLaunch?.(route.request().postDataJSON());
    const next = launchStatuses.shift() ?? awaitingReviewStatus();
    if (typeof next.status === "number") {
      await route.fulfill({
        status: next.status,
        contentType: "application/json",
        json: { detail: next.code ? { code: next.code } : "lost response" },
      });
      return;
    }
    await route.fulfill({ contentType: "application/json", json: next });
  });
}

function awaitingReviewStatus() {
  return {
    thread_id: "workflow-thread-1",
    status: "awaiting_human_review",
    review_item_count: 2,
    review_status_counts: { pending_review: 2 },
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
}

async function completeGmailSync(page: Page) {
  await page.addInitScript(() => window.localStorage.setItem("paraworks-demo-user", "demo-admin"));
  await page.goto("/integrations");
  await page.getByTestId("gmail-card-actions").getByRole("button", { name: "동기화" }).click();
  await expect(page.getByTestId("sync-progress-modal")).toBeVisible();
}

test("shows cost and launches the exact completed source batch", async ({ page }) => {
  let dryRunBody: unknown;
  let launchBody: unknown;
  await installIntegrationsRoutes(page, {
    onDryRun: (body) => {
      dryRunBody = body;
    },
    onLaunch: (body) => {
      launchBody = body;
    },
  });

  await completeGmailSync(page);

  const panel = page.getByTestId("review-candidate-launch-panel");
  const modal = page.getByTestId("sync-progress-modal");
  await expect(panel).toContainText("예상 비용");
  await expect(panel).toContainText("예산 이내");
  await expect(panel).toContainText("120");
  await expect(panel).toContainText("메모리 모드");
  await expect(modal.getByTestId("sync-modal-step")).toContainText("변경 근거를 준비했습니다");
  await expect(modal).not.toContainText("검토 큐에 반영했습니다");
  await expect(modal).not.toContainText("AI 분석");
  await expect(modal).not.toContainText("검토 항목 저장");
  expect(dryRunBody).toEqual({ source_refs: sourceRefs, agent_names: diagnostic.default_agent_names });

  await panel.getByRole("button", { name: "검토 후보 만들기" }).click();
  await expect(page).toHaveURL(/\/review\?workflow_thread_id=workflow-thread-1$/);
  expect(launchBody).toEqual({
    source_refs: sourceRefs,
    agent_names: diagnostic.default_agent_names,
    client_request_id: "review:gmail-sync-review-v2:company-memory-review-selection:v1",
  });
});

test("uses canonical refs returned by queued runtime completion", async ({ page }) => {
  let dryRunBody: unknown;
  await installIntegrationsRoutes(page, {
    syncResponses: [
      {
        ...completedSync,
        status: "queued",
        changed_source_refs: [],
      },
    ],
    runtimeSyncResponses: [
      {
        job_id: completedSync.job_id,
        status: "complete",
        message: "fetched=2 created_review_items=0 skipped_events=0 pending_review_items=3",
        progress_pct: 100,
        changed_source_refs: sourceRefs,
      },
    ],
    onDryRun: (body) => {
      dryRunBody = body;
    },
  });

  await completeGmailSync(page);

  await expect(page.getByTestId("review-candidate-launch-panel")).toContainText("검토 후보 미리보기");
  expect(dryRunBody).toEqual({ source_refs: sourceRefs, agent_names: diagnostic.default_agent_names });
});

test("keeps the user on integrations when no candidates are created", async ({ page }) => {
  await installIntegrationsRoutes(page, {
    launchStatuses: [{ ...awaitingReviewStatus(), status: "completed", review_item_count: 0 }],
  });

  await completeGmailSync(page);
  await page.getByTestId("review-candidate-launch-panel").getByRole("button", { name: "검토 후보 만들기" }).click();

  await expect(page).toHaveURL(/\/integrations$/);
  await expect(page.getByTestId("review-candidate-launch-panel")).toContainText("새 검토 후보 없음");
});

test("hides launch when readiness is disabled or unavailable", async ({ page }) => {
  await installIntegrationsRoutes(page, {
    diagnosticResponse: { ...diagnostic, enabled: false, available: false, checkpoint_mode: "disabled" },
  });

  await completeGmailSync(page);

  await expect(page.getByTestId("review-candidate-launch-panel")).toHaveCount(0);
  await expect(page.getByTestId("sync-modal-step")).toContainText("검토 큐에 반영했습니다");
  await expect(page.getByTestId("sync-progress-modal")).toContainText("AI 분석");
  await expect(page.getByTestId("sync-progress-modal")).toContainText("검토 항목 저장");
});

test("does not promise a V2 action when the completed batch has zero canonical refs", async ({ page }) => {
  await installIntegrationsRoutes(page, {
    syncResponses: [{ ...completedSync, changed_source_refs: [] }],
  });

  await completeGmailSync(page);
  const modal = page.getByTestId("sync-progress-modal");
  await expect(page.getByTestId("review-candidate-launch-panel")).toHaveCount(0);
  await expect(modal.getByTestId("sync-modal-step")).toContainText("변경 근거를 찾지 못했습니다");
  await expect(modal).not.toContainText("아래에서 명시적으로 만듭니다");
  await expect(modal).not.toContainText("검토 큐에 반영했습니다");
  await expect(modal).not.toContainText("AI 분석");
});

test("shows bounded unavailable guidance when the V2 diagnostic fails permanently", async ({ page }) => {
  await installIntegrationsRoutes(page, { diagnosticFailure: true });

  await completeGmailSync(page);
  const modal = page.getByTestId("sync-progress-modal");
  await expect(page.getByTestId("review-candidate-launch-panel")).toHaveCount(0);
  await expect(modal.getByTestId("sync-modal-step")).toContainText("검토 후보 기능을 확인할 수 없습니다");
  await expect(modal).not.toContainText("아래에서 명시적으로 만듭니다");
  await expect(modal).not.toContainText("검토 큐에 반영했습니다");
  await expect(modal).not.toContainText("AI 분석");
});

test("keeps completion copy truthful while the V2 diagnostic is still resolving", async ({ page }) => {
  let resolveDiagnostic: (() => void) | undefined;
  const diagnosticGate = new Promise<void>((resolve) => {
    resolveDiagnostic = resolve;
  });
  await installIntegrationsRoutes(page);
  await page.route("**/api/v1/orchestration/v2/company-memory", async (route) => {
    await diagnosticGate;
    await route.fulfill({ contentType: "application/json", json: diagnostic });
  });

  await completeGmailSync(page);
  const modal = page.getByTestId("sync-progress-modal");
  await expect(modal.getByTestId("sync-modal-step")).toContainText("검토 후보 가능 여부를 확인하고 있습니다");
  await expect(modal).not.toContainText("검토 큐에 반영했습니다");
  await expect(modal).not.toContainText("아래에서 명시적으로 만듭니다");
  resolveDiagnostic?.();
  await expect(page.getByTestId("review-candidate-launch-panel")).toBeVisible();
});

test("does not offer V2 launch for Slack sync", async ({ page }) => {
  await installIntegrationsRoutes(page, { connectorType: "slack" });
  await page.addInitScript(() => window.localStorage.setItem("paraworks-demo-user", "demo-admin"));
  await page.goto("/integrations");
  await page.getByTestId("slack-card-actions").getByRole("button", { name: "동기화" }).click();

  await expect(page.getByTestId("sync-progress-modal")).toBeVisible();
  await expect(page.getByTestId("review-candidate-launch-panel")).toHaveCount(0);
});

test("reuses one stable client request id after response loss", async ({ page }) => {
  const launchBodies: unknown[] = [];
  await installIntegrationsRoutes(page, {
    launchStatuses: [{ status: 500 }, awaitingReviewStatus()],
    onLaunch: (body) => launchBodies.push(body),
  });

  await completeGmailSync(page);
  const button = page.getByTestId("review-candidate-launch-panel").getByRole("button", { name: "검토 후보 만들기" });
  await button.click();
  await expect(page.getByTestId("review-candidate-launch-panel")).toContainText("다시 시도");
  await button.click();

  await expect(page).toHaveURL(/\/review\?workflow_thread_id=workflow-thread-1$/);
  expect(launchBodies).toHaveLength(2);
  expect(launchBodies[0]).toEqual(launchBodies[1]);
});

for (const [code, guidance] of [
  ["evidence_changed", "데이터 변경 후 다시 동기화"],
  ["idempotency_key_reused", "새 미리보기를 준비"],
  ["budget_exceeded", "새 미리보기를 준비"],
] as const) {
  test(`does not replay a stable launch id after non-retryable ${code}`, async ({ page }) => {
    const launchBodies: unknown[] = [];
    await installIntegrationsRoutes(page, {
      launchStatuses: [{ status: 409, code }],
      onLaunch: (body) => launchBodies.push(body),
    });

    await completeGmailSync(page);
    const panel = page.getByTestId("review-candidate-launch-panel");
    const button = panel.getByRole("button", { name: "검토 후보 만들기" });
    await button.click();

    await expect(panel).toContainText(guidance);
    await expect(panel).not.toContainText(code);
    await expect(button).toBeDisabled();
    expect(launchBodies).toEqual([
      {
        source_refs: sourceRefs,
        agent_names: diagnostic.default_agent_names,
        client_request_id: "review:gmail-sync-review-v2:company-memory-review-selection:v1",
      },
    ]);
  });
}

test("does not navigate when a delayed launch belongs to a superseded sync", async ({ page }) => {
  let signalLaunchStarted: (() => void) | undefined;
  const launchStarted = new Promise<void>((resolve) => {
    signalLaunchStarted = resolve;
  });
  let releaseLaunch: (() => void) | undefined;
  const waitForRelease = new Promise<void>((resolve) => {
    releaseLaunch = resolve;
  });
  const newerRefs = [
    {
      source_type: "calendar",
      source_id: "calendar:event-new",
      version_or_signature: "calendar:v2",
    },
  ];
  const dryRunBodies: unknown[] = [];
  await installIntegrationsRoutes(page, {
    syncResponses: [
      { ...completedSync, job_id: "gmail-sync-old", changed_source_refs: sourceRefs },
      { ...completedSync, job_id: "gmail-sync-new", changed_source_refs: newerRefs },
    ],
    dryRunResponses: [dryRun, { ...dryRun, source_count: newerRefs.length }],
    onDryRun: (body) => dryRunBodies.push(body),
  });
  await page.route("**/api/v1/orchestration/v2/company-memory/runs", async (route) => {
    signalLaunchStarted?.();
    await waitForRelease;
    await route.fulfill({ contentType: "application/json", json: awaitingReviewStatus() });
  });

  await completeGmailSync(page);
  const firstLaunch = page.getByTestId("review-candidate-launch-panel").getByRole("button", { name: "검토 후보 만들기" }).click();
  await launchStarted;
  await page.getByRole("button", { name: "모달 닫기" }).click();
  await page.getByTestId("gmail-card-actions").getByRole("button", { name: "동기화" }).click();
  await expect(page.getByTestId("review-candidate-launch-panel")).toContainText("1개");
  releaseLaunch?.();
  await firstLaunch;

  await expect(page).toHaveURL(/\/integrations$/);
  expect(dryRunBodies).toEqual([
    { source_refs: sourceRefs, agent_names: diagnostic.default_agent_names },
    { source_refs: newerRefs, agent_names: diagnostic.default_agent_names },
  ]);
});

test("renders over-budget cost outcome and blocks launch", async ({ page }) => {
  await installIntegrationsRoutes(page, {
    dryRunResponses: [{ ...dryRun, budget_status: "over_budget" }],
  });

  await completeGmailSync(page);
  const panel = page.getByTestId("review-candidate-launch-panel");
  await expect(panel).toContainText("예산 초과");
  await expect(panel.getByRole("button", { name: "검토 후보 만들기" })).toBeDisabled();
});

test("renders cached cost outcome", async ({ page }) => {
  await installIntegrationsRoutes(page, {
    dryRunResponses: [{ ...dryRun, budget_status: "cached", cache_hit: true }],
  });

  await completeGmailSync(page);
  await expect(page.getByTestId("review-candidate-launch-panel")).toContainText("캐시 재사용 · 추가 모델 호출 없음");
});

test("renders no-input cost outcome", async ({ page }) => {
  await installIntegrationsRoutes(page, {
    dryRunResponses: [{ ...dryRun, budget_status: "no_input", estimated_cost_usd: 0 }],
  });

  await completeGmailSync(page);
  await expect(page.getByTestId("review-candidate-launch-panel")).toContainText("추가 비용 없음");
});

test("offers a working preview retry after transient dry-run failure", async ({ page }) => {
  await installIntegrationsRoutes(page, {
    dryRunResponses: [{ status: 500 }, dryRun],
  });

  await completeGmailSync(page);
  const panel = page.getByTestId("review-candidate-launch-panel");
  await expect(panel).toContainText("비용을 확인하지 못했습니다");
  await panel.getByRole("button", { name: "미리보기 다시 시도" }).click();
  await expect(panel).toContainText("예산 이내");
});

test("preserves failed terminal status and blocks its stable-id replay", async ({ page }) => {
  await installIntegrationsRoutes(page, {
    launchStatuses: [{ ...awaitingReviewStatus(), status: "failed", review_item_count: 0 }],
  });

  await completeGmailSync(page);
  const button = page.getByTestId("review-candidate-launch-panel").getByRole("button", { name: "검토 후보 만들기" });
  await button.click();

  const panel = page.getByTestId("review-candidate-launch-panel");
  await expect(panel).not.toContainText("새 검토 후보 없음");
  await expect(panel).toContainText("작업 실패");
  await expect(panel).toContainText("데이터 변경 후 다시 동기화");
  await expect(button).toBeDisabled();
});

test("preserves cancelled terminal status and blocks its stable-id replay", async ({ page }) => {
  await installIntegrationsRoutes(page, {
    launchStatuses: [{ ...awaitingReviewStatus(), status: "cancelled", review_item_count: 0 }],
  });

  await completeGmailSync(page);
  const button = page.getByTestId("review-candidate-launch-panel").getByRole("button", { name: "검토 후보 만들기" });
  await button.click();

  const panel = page.getByTestId("review-candidate-launch-panel");
  await expect(panel).not.toContainText("새 검토 후보 없음");
  await expect(panel).toContainText("작업 취소됨");
  await expect(panel).toContainText("데이터 변경 후 다시 동기화");
  await expect(button).toBeDisabled();
});
