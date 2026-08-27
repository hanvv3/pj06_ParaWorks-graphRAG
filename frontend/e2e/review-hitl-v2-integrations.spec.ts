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
    launchStatuses?: Array<Record<string, unknown>>;
    onDryRun?: (body: unknown) => void;
    onLaunch?: (body: unknown) => void;
  } = {},
) {
  const connectorType = options.connectorType ?? "gmail";
  const diagnosticResponse = options.diagnosticResponse ?? diagnostic;
  const launchStatuses = [...(options.launchStatuses ?? [awaitingReviewStatus()])];

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
      await route.fulfill({
        contentType: "application/json",
        json: {
          connector_type: type,
          mode: "live",
          connection_status: type === connectorType ? "connected" : "disconnected",
          credential_status: type === connectorType ? "available" : "missing",
          latest_sync: null,
          cost_policy: { status_lookup_triggers_sync: false, status_lookup_triggers_llm: false },
        },
      });
    });
  }
  await page.route(`**/api/v1/integrations/${connectorType}/sync`, async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: { ...completedSync, connector_type: connectorType },
    });
  });
  await page.route("**/api/v1/orchestration/v2/company-memory", async (route) => {
    await route.fulfill({ contentType: "application/json", json: diagnosticResponse });
  });
  await page.route("**/api/v1/orchestration/v2/company-memory/dry-run", async (route) => {
    options.onDryRun?.(route.request().postDataJSON());
    await route.fulfill({ contentType: "application/json", json: dryRun });
  });
  await page.route("**/api/v1/orchestration/v2/company-memory/runs", async (route) => {
    options.onLaunch?.(route.request().postDataJSON());
    const next = launchStatuses.shift() ?? awaitingReviewStatus();
    if (next.status === 500) {
      await route.fulfill({ status: 500, contentType: "application/json", json: { detail: "lost response" } });
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
  await expect(panel).toContainText("예상 비용");
  await expect(panel).toContainText("120");
  await expect(panel).toContainText("메모리 모드");
  expect(dryRunBody).toEqual({ source_refs: sourceRefs, agent_names: diagnostic.default_agent_names });

  await panel.getByRole("button", { name: "검토 후보 만들기" }).click();
  await expect(page).toHaveURL(/\/review\?workflow_thread_id=workflow-thread-1$/);
  expect(launchBody).toEqual({
    source_refs: sourceRefs,
    agent_names: diagnostic.default_agent_names,
    client_request_id: "review:gmail-sync-review-v2:company-memory-review-selection:v1",
  });
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

test("does not label failed or cancelled zero effect replay as no candidates", async ({ page }) => {
  await installIntegrationsRoutes(page, {
    launchStatuses: [
      { ...awaitingReviewStatus(), status: "failed", review_item_count: 0 },
      { ...awaitingReviewStatus(), status: "cancelled", review_item_count: 0 },
    ],
  });

  await completeGmailSync(page);
  const button = page.getByTestId("review-candidate-launch-panel").getByRole("button", { name: "검토 후보 만들기" });
  await button.click();

  const panel = page.getByTestId("review-candidate-launch-panel");
  await expect(panel).not.toContainText("새 검토 후보 없음");
  await expect(panel).toContainText("완료되지 않았습니다");
  await button.click();
  await expect(panel).not.toContainText("새 검토 후보 없음");
  await expect(panel).toContainText("완료되지 않았습니다");
});
