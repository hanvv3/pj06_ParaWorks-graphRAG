import { expect, request, test, type Page } from "@playwright/test";

const backendBaseURL = process.env.PLAYWRIGHT_API_BASE_URL ?? "http://127.0.0.1:8000";
const pages = [
  { path: "/dashboard", heading: "오늘의 업무 흐름" },
  { path: "/messages", heading: "메신저" },
  { path: "/review", heading: "검토 큐" },
  { path: "/knowledge", heading: "승인된 회사 메모리" },
  { path: "/integrations", heading: "연동과 에이전트 도구" },
  { path: "/agent-runs", heading: "AI 실행 관측" },
  { path: "/search", heading: "회사 메모리에 질문하기" },
  { path: "/login", heading: "로그인" },
  { path: "/admin", heading: "관리자 콘솔" },
];

test.beforeAll(async () => {
  if (process.env.PLAYWRIGHT_SKIP_BACKEND_SEED === "1") return;

  const api = await request.newContext({ baseURL: backendBaseURL });
  try {
    await api.post("/api/v1/integrations/slack/sync");
    await api.post("/api/v1/integrations/gmail/sync");
    await api.post("/api/v1/rag/reindex/jobs");
  } finally {
    await api.dispose();
  }
});

async function mockTask20ShellAndAssistantApis(page: Page) {
  await page.route("**/api/v1/**", (route) => route.fulfill({
    contentType: "application/json",
    json: {},
  }));
  await page.route("**/api/v1/auth/me", (route) => route.fulfill({
    contentType: "application/json",
    json: {
      user: {
        id: "reviewer-task20",
        email: "reviewer@paraworks.test",
        role: "reviewer",
        permission_levels: ["internal"],
        name: "Task20 Reviewer",
        title: "Reviewer",
        department: "Product",
      },
    },
  }));
  await page.route("**/api/v1/dashboard", (route) => route.fulfill({
    contentType: "application/json",
    json: {
      source_counts: {},
      pending_review_count: 0,
      recent_jobs: [],
      pending_items: [],
      today_todos: [],
      recent_decisions: [],
      recent_timeline: [],
    },
  }));
  await page.route("**/api/v1/notifications", (route) => route.fulfill({
    contentType: "application/json",
    json: { notifications: [], counts: { total: 0 } },
  }));
  await page.route("**/api/v1/integrations", (route) => route.fulfill({
    contentType: "application/json",
    json: [],
  }));
  await page.route("**/api/v1/integrations/connections", (route) => route.fulfill({
    contentType: "application/json",
    json: [],
  }));
  await page.route("**/api/v1/assistant/conversations", (route) => route.fulfill({
    contentType: "application/json",
    json: {
      conversations: [{
        id: 90,
        title: "새 대화",
        summary: null,
        created_at: "2026-09-13T00:00:00Z",
        updated_at: "2026-09-13T00:00:00Z",
      }],
    },
  }));
  await page.route("**/api/v1/assistant/conversations/90/messages", (route) => route.fulfill({
    contentType: "application/json",
    json: {
      conversation: {
        id: 90,
        title: "새 대화",
        summary: null,
        created_at: "2026-09-13T00:00:00Z",
        updated_at: "2026-09-13T00:00:00Z",
      },
      messages: [],
    },
  }));
}

for (const target of pages) {
  test(`${target.path} renders Korean workspace UI without mojibake`, async ({ page }) => {
    await page.goto(target.path);
    await expect(page.getByRole("heading", { name: target.heading })).toBeVisible();

    const bodyText = await page.locator("body").innerText();
    expect(bodyText).not.toContain("�");
    expect(bodyText).not.toContain("?꾩");
    expect(bodyText).not.toContain("?ㅽ");
    expect(bodyText).not.toContain("?덉");
    expect(bodyText).not.toContain("Application error");
    expect(bodyText).not.toContain('"detail":"Not Found"');
    expect(bodyText).not.toContain("{\"detail\"");
  });
}

test("integrations page keeps all connector cards when OAuth status endpoints are optional", async ({ page }) => {
  await page.goto("/integrations");

  await expect(page.getByRole("heading", { name: "Slack" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Gmail" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Google Drive" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Google Calendar" })).toBeVisible();
});

test("theme toggle switches between dark and light glass modes", async ({ page }) => {
  await page.goto("/dashboard");
  await expect(page.getByTestId("app-shell")).toHaveAttribute("data-hydrated", "true");

  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.getByRole("button", { name: "라이트 모드" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.getByRole("button", { name: "다크 모드" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
});

test("workspace search hands input to the company memory page without a query URL", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name.includes("mobile"), "desktop sidebar search is hidden on mobile");
  await mockTask20ShellAndAssistantApis(page);

  const authResponse = page.waitForResponse("**/api/v1/auth/me");
  await page.goto("/integrations");
  await authResponse;
  const shellSearch = page.getByRole("textbox", { name: "회사 메모리 검색" });
  await shellSearch.fill("Redis queue state");
  await shellSearch.press("Enter");

  await expect(page).toHaveURL(/\/search$/);
  await expect(page.getByRole("textbox", { name: "AI 비서에게 질문" })).toHaveValue("Redis queue state");
});

test("top search preserves raw input only in the same-screen ephemeral handoff", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name.includes("mobile"), "desktop top search is hidden on mobile");
  await mockTask20ShellAndAssistantApis(page);

  const authResponse = page.waitForResponse("**/api/v1/auth/me");
  await page.goto("/dashboard");
  await authResponse;
  const shellSearch = page.getByRole("textbox", { name: "회사 메모리 검색" });
  await shellSearch.fill("  PostgreSQL durable record  ");
  await shellSearch.press("Enter");

  await expect(page).toHaveURL(/\/search$/);
  await expect(page.getByRole("textbox", { name: "AI 비서에게 질문" })).toHaveValue("  PostgreSQL durable record  ");
});

test("demo login switches the active API user", async ({ page }) => {
  await page.goto("/login");

  await expect(page.getByRole("heading", { name: "로그인" })).toBeVisible();
  await page.getByRole("button", { name: "이 계정으로 로그인" }).first().click();
  await expect(page.getByText("ParaWorks Admin 계정으로 전환되었습니다.")).toBeVisible();

  const storedUser = await page.evaluate(() => window.localStorage.getItem("paraworks-demo-user"));
  expect(storedUser).toBe("demo-admin");
});

test("login pending state centers loading affordance over dimmed page", async ({ page }) => {
  let releaseLogin!: () => void;
  const loginPending = new Promise<void>((resolve) => {
    releaseLogin = resolve;
  });

  await page.route("**/api/v1/auth/login", async (route) => {
    await loginPending;
    await route.fulfill({
      contentType: "application/json",
      json: {
        user: {
          id: "demo-admin",
          email: "admin@paraworks.com",
          name: "ParaWorks Admin",
          role: "admin",
          department: "Operations",
        },
      },
    });
  });

  await page.goto("/login");
  await page.getByTestId("login-submit").click();

  const overlay = page.getByTestId("login-loading-overlay");
  const card = page.getByTestId("login-loading-card");

  await expect(overlay).toBeVisible();
  await expect(overlay).toHaveCSS("position", "fixed");
  await expect(overlay).toHaveCSS("background-color", "rgba(15, 23, 42, 0.72)");

  const cardBox = await card.boundingBox();
  const viewport = page.viewportSize();
  expect(cardBox).not.toBeNull();
  expect(viewport).not.toBeNull();

  if (cardBox && viewport) {
    const cardCenterX = cardBox.x + cardBox.width / 2;
    const cardCenterY = cardBox.y + cardBox.height / 2;
    expect(Math.abs(cardCenterX - viewport.width / 2)).toBeLessThan(4);
    expect(Math.abs(cardCenterY - viewport.height / 2)).toBeLessThan(4);
  }

  releaseLogin();
});

test("google login callback centers loading state", async ({ page }) => {
  let releaseCallback!: () => void;
  const callbackPending = new Promise<void>((resolve) => {
    releaseCallback = resolve;
  });

  await page.route("**/api/v1/auth/google/callback?**", async (route) => {
    await callbackPending;
    await route.fulfill({
      contentType: "application/json",
      json: {
        user: {
          id: "google-hanvv-employee",
          email: "hanvv3@gmail.com",
          name: "한승헌",
          role: "employee",
          department: "Operations",
        },
      },
    });
  });

  await page.goto("/login/google/callback?code=test-code&state=test-state");

  const pageShell = page.getByTestId("google-login-callback-page");
  const statusCard = page.getByTestId("google-login-callback-status");

  await expect(pageShell).toBeVisible();
  await expect(statusCard).toBeVisible();
  await expect(statusCard).toContainText("Google 로그인 결과를 확인하고 있습니다.");

  const shellBox = await pageShell.boundingBox();
  const cardBox = await statusCard.boundingBox();
  const viewport = page.viewportSize();
  expect(shellBox).not.toBeNull();
  expect(cardBox).not.toBeNull();
  expect(viewport).not.toBeNull();

  if (shellBox && viewport) {
    expect(shellBox.height).toBeGreaterThanOrEqual(viewport.height);
  }

  if (cardBox && viewport) {
    const cardCenterX = cardBox.x + cardBox.width / 2;
    const cardCenterY = cardBox.y + cardBox.height / 2;
    expect(Math.abs(cardCenterX - viewport.width / 2)).toBeLessThan(6);
    expect(Math.abs(cardCenterY - viewport.height / 2)).toBeLessThan(6);
  }

  releaseCallback();
});

test("admin console is blocked for employee accounts", async ({ page }) => {
  await page.addInitScript(() => window.localStorage.setItem("paraworks-demo-user", "google-hanvv-employee"));
  await page.goto("/admin");

  await expect(page.getByRole("heading", { name: "관리자 권한 필요" })).toBeVisible();
  await expect(page.getByRole("link", { name: "관리자 계정으로 로그인" })).toBeVisible();
});

test("admin console lists demo employees and permission levels", async ({ page }) => {
  await page.addInitScript(() => window.localStorage.setItem("paraworks-demo-user", "demo-admin"));
  await page.goto("/admin");

  await expect(page.getByRole("heading", { name: "관리자 콘솔" })).toBeVisible();
  await expect(page.getByText("admin@paraworks.com").first()).toBeVisible();
  await expect(page.getByText("kjw4work@gmail.com")).toBeVisible();
  await expect(page.getByText("yonghee199702@gmail.com")).toBeVisible();
  await expect(page.getByText("mina@paraworks.com")).toBeVisible();
  await expect(page.getByText("jun@paraworks.com")).toHaveCount(0);
  await expect(page.getByText("soyeon@paraworks.com")).toHaveCount(0);
  await expect(page.getByText("restricted").first()).toBeVisible();
});

test("agent operations previews RAG reindex cost before approved execution", async ({ page }) => {
  await page.route("**/api/v1/rag/reindex**", async (route) => {
    const url = new URL(route.request().url());
    if (url.searchParams.get("dry_run") !== "true") {
      await route.fallback();
      return;
    }
    await route.fulfill({
      contentType: "application/json",
      json: {
        dry_run: true,
        indexed_count: 2,
        skipped_count: 3,
        saved_embedding_calls: 3,
        embedding_request_count: 1,
        embedding_prompt_tokens: 0,
        embedding_total_tokens: 0,
        embedding_dimensions: 16,
        document_ids: ["chunk:1", "decision_record:2"],
        skipped_document_ids: ["chunk:old"],
        incremental: true,
        storage_backend: "preview",
        parser_status_counts: {
          parsed: 1,
          metadata_only: 1,
        },
        embedding_budget: {
          embedding_model: "text-embedding-3-small",
          changed_document_count: 2,
          estimated_input_tokens: 1200,
          estimated_cost_usd: 0.000024,
          budget_limit_usd: 0.001,
          budget_status: "within_budget",
          action: "run",
          reason: "within_embedding_budget",
        },
      },
    });
  });
  await page.route("**/api/v1/rag/reindex/jobs**", async (route) => {
    const url = new URL(route.request().url());
    if (url.searchParams.get("dry_run") !== "false") {
      await route.fallback();
      return;
    }
    await route.fulfill({
      contentType: "application/json",
      json: {
        job_id: "rag-index-approved",
        status: "queued",
        dry_run: false,
      },
    });
  });

  await page.goto("/agent-runs");
  await expect(page.getByTestId("app-shell")).toHaveAttribute("data-hydrated", "true");
  await expect(page.getByTestId("rag-reindex-control")).toBeVisible();
  await page.getByRole("button", { name: "비용 미리보기" }).click();
  await expect(page.getByTestId("rag-reindex-preview")).toContainText("변경 2개");
  await expect(page.getByTestId("rag-reindex-preview")).toContainText("$0.000024");
  await expect(page.getByTestId("rag-parser-quality")).toContainText("Parser quality");
  await expect(page.getByTestId("rag-parser-quality")).toContainText("Metadata only");
  await page.getByRole("button", { name: "승인 후 실행" }).click();
  await expect(page.getByText("rag-index-approved")).toBeVisible();
});

test("shell chrome uses distinct theme tokens across viewport modes", async ({ page }, testInfo) => {
  const isMobile = testInfo.project.name.includes("mobile");
  const shellSelector = isMobile ? "header.md\\:hidden .liquid-surface" : "aside.shell-rail";

  await page.addInitScript((theme) => window.localStorage.setItem("paraworks-theme", theme), "dark");
  await page.goto("/integrations");
  await expect(page.getByTestId("app-shell")).toHaveAttribute("data-hydrated", "true");

  const darkStyle = await page.locator(shellSelector).evaluate((element) => {
    const style = window.getComputedStyle(element);
    return {
      color: style.color,
      backgroundColor: style.backgroundColor,
      borderColor: style.borderColor,
    };
  });

  await page.getByRole("button", { name: "라이트 모드" }).click();

  const lightStyle = await page.locator(shellSelector).evaluate((element) => {
    const style = window.getComputedStyle(element);
    return {
      color: style.color,
      backgroundColor: style.backgroundColor,
      borderColor: style.borderColor,
    };
  });

  if (!isMobile) {
    expect(lightStyle.color).not.toBe(darkStyle.color);
  }
  expect(lightStyle.backgroundColor).not.toBe(darkStyle.backgroundColor);
  expect(lightStyle.borderColor).not.toBe(darkStyle.borderColor);

  if (!isMobile) {
    const inactiveLinkColor = await page.locator("aside nav a:not(.liquid-segment-active)").first().evaluate((element) => {
      return window.getComputedStyle(element).color;
    });
    expect(inactiveLinkColor).not.toBe("rgba(255, 255, 255, 0.7)");
  }
});

test("integration sync shows connector counts", async ({ page }) => {
  await page.goto("/integrations");
  await expect(page.getByRole("heading", { name: "연동과 에이전트 도구" })).toBeVisible();
  await page.getByRole("button", { name: "동기화" }).first().click();

  const sourcePanel = page.getByTestId("source-operations-panel");
  await expect(sourcePanel).toBeVisible();
  await expect(page.getByTestId("source-operation-slack-header")).toContainText("Slack");
  await expect(page.getByTestId("source-operation-slack-count")).toBeVisible();
  await expect(page.getByTestId("source-operation-slack-bar")).toBeVisible();
  await expect(page.getByTestId("source-operation-other")).toHaveCount(0);
  await expect(sourcePanel).not.toContainText("기타");
  await expect
    .poll(async () => {
      const header = await page.getByTestId("source-operation-slack-header").boundingBox();
      const bar = await page.getByTestId("source-operation-slack-bar").boundingBox();
      if (!header || !bar) return false;
      return bar.y > header.y + header.height - 1;
    })
    .toBe(true);
  await expect(sourcePanel).not.toContainText("%");
});

test("Gmail source status keeps count separate from the progress bar", async ({ page }) => {
  await page.route("**/api/v1/integrations/gmail/sync", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: {
        job_id: "gmail-parser-quality-smoke",
        connector_type: "gmail",
        status: "complete",
        created_review_items: 1,
        fetched_events: 2,
        skipped_events: 0,
        parser_status_counts: {
          metadata_only: 1,
        },
      },
    });
  });

  await page.goto("/integrations");
  await expect(page.getByRole("heading", { name: "연동과 에이전트 도구" })).toBeVisible();
  await page.getByTestId("gmail-card-actions").getByRole("button").first().click();

  await expect(page.getByTestId("source-operation-gmail-header")).toContainText("Gmail");
  await expect(page.getByTestId("source-operation-gmail-count")).toBeVisible();
  await expect(page.getByTestId("source-operation-gmail-bar")).toBeVisible();
  await expect(page.getByTestId("source-operations-panel")).not.toContainText("%");
});

test("integrations page shows Slack OAuth connection status without secrets", async ({ page }) => {
  await page.goto("/integrations");

  await expect(page.locator('[data-testid="slack-oauth-status"]')).toBeVisible();
  await expect(page.getByTestId("slack-oauth-status")).not.toContainText("ready");
  await expect(page.getByTestId("slack-runtime-status")).toHaveCount(0);

  const bodyText = await page.locator("body").innerText();
  expect(bodyText).not.toContain("xoxb-");
  expect(bodyText).not.toContain("client-secret");
  expect(bodyText).not.toContain("token_ref");
  expect(bodyText).not.toContain("실제 OAuth 연동");
});

test("Slack card keeps OAuth install outside primary action row", async ({ page }) => {
  await page.goto("/integrations");

  const slackActions = page.getByTestId("slack-card-actions");
  await expect(slackActions.getByRole("button", { name: "동기화" })).toHaveCount(0);
  await expect(slackActions.getByRole("button", { name: "Slack Agent 실행" })).toHaveCount(0);
  await expect(slackActions.getByRole("button", { name: "Slack 연결" })).toHaveCount(0);
});

test("integration connector cards use consistent chrome and action layout", async ({ page }) => {
  await page.goto("/integrations");
  await expect(page.getByRole("heading", { name: "연동과 에이전트 도구" })).toBeVisible();

  const cards = page.locator("article.integration-glass-card");
  await expect(cards).toHaveCount(4);
  await expect(cards.filter({ has: page.getByRole("heading", { name: "Slack" }) })).not.toContainText("우선순위");
  for (const connectorType of ["calendar", "drive", "gmail", "slack"]) {
    const logo = page.getByTestId(`${connectorType}-connector-logo`);
    await expect(logo).toBeVisible();
    const logoSvg = logo.locator("svg");
    await expect(logoSvg).toBeVisible();
    await expect(logoSvg).toHaveAttribute("data-logo-source", /official/);
  }

  const borderColors = await cards.evaluateAll((elements) =>
    elements.map((element) => window.getComputedStyle(element).borderColor),
  );
  expect(new Set(borderColors).size).toBe(1);

  const heights = await cards.evaluateAll((elements) =>
    elements.map((element) => Math.round(element.getBoundingClientRect().height)),
  );
  expect(Math.max(...heights) - Math.min(...heights)).toBeLessThanOrEqual(1);
  expect(Math.max(...heights)).toBeLessThanOrEqual(400);

  const actionHeights = await page.locator('[data-testid$="-card-actions"]').evaluateAll((elements) =>
    elements.map((element) => Math.round(element.getBoundingClientRect().height)),
  );
  expect(Math.max(...actionHeights)).toBeLessThanOrEqual(46);

  const actionDisplays = await page.locator('[data-testid$="-card-actions"]').evaluateAll((elements) =>
    elements.map((element) => window.getComputedStyle(element).display),
  );
  expect(new Set(actionDisplays)).toEqual(new Set(["flex"]));

  const actionGaps = await page.evaluate(() =>
    ["calendar", "drive", "gmail", "slack"].map((type) => {
      const oauthStatus = document.querySelector(`[data-testid="${type}-oauth-status"]`);
      const actions = document.querySelector(`[data-testid="${type}-card-actions"]`);
      if (!oauthStatus || !actions) return 0;
      return Math.round(actions.getBoundingClientRect().top - oauthStatus.getBoundingClientRect().bottom);
    }),
  );
  expect(Math.max(...actionGaps)).toBeLessThanOrEqual(28);

  const driveActionLabels = await page.getByTestId("drive-card-actions").locator("button, a").evaluateAll((elements) =>
    elements.map((element) => element.textContent?.replace(/\s+/g, " ").trim()),
  );
  expect(driveActionLabels).toContain("동기화");
  expect(driveActionLabels).toContain("문서 현황");
});

test("Slack OAuth callback route renders a safe local error without secrets", async ({ page }) => {
  await page.goto("/integrations/slack/callback");

  await expect(page.getByRole("heading", { name: "Slack 연결 확인" })).toBeVisible();
  await expect(page.getByText("Slack 연결 정보를 확인할 수 없습니다.")).toBeVisible();

  const bodyText = await page.locator("body").innerText();
  expect(bodyText).not.toContain("xoxb-");
  expect(bodyText).not.toContain("client-secret");
  expect(bodyText).not.toContain("token_ref");
});

test("Slack OAuth status shows reconnect CTA when the local credential is missing", async ({ page }) => {
  await page.route("**/api/v1/integrations/connections", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: [
        {
          connector_type: "slack",
          workspace_id: "T123",
          workspace_name: "ParaWorks Demo",
          status: "connected",
          credential_status: "missing",
          masked_bot_token: "xoxb...demo",
          scopes: ["channels:history"],
        },
      ],
    });
  });
  await page.route("**/api/v1/integrations/slack/oauth/install-url", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: {
        connector_type: "slack",
        configured: true,
        install_url: "https://slack.com/oauth/v2/authorize?client_id=C123",
        state: "signed-state",
        required_scopes: ["channels:history"],
      },
    });
  });

  await page.goto("/integrations");

  await expect(page.getByTestId("slack-oauth-workspace-name")).toHaveText("ParaWorks Demo");
  await expect(page.getByTestId("slack-oauth-workspace-name")).toHaveCSS("white-space", "nowrap");
  await expect(page.getByText("ParaWorks Demo 재연결 필요")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Slack 재연결" })).toBeVisible();
  await expect(page.getByTestId("slack-card-actions").getByRole("button", { name: "Slack 재연결" })).toHaveCount(0);
});

test("Google connector cards show OAuth readiness outside primary action rows", async ({ page }) => {
  await page.goto("/integrations");

  const connectors = [
    { type: "gmail", label: "Gmail" },
    { type: "drive", label: "Google Drive" },
    { type: "calendar", label: "Google Calendar" },
  ];

  for (const connector of connectors) {
    const oauthStatus = page.getByTestId(`${connector.type}-oauth-status`);
    await expect(oauthStatus).toBeVisible();
    await expect(oauthStatus).not.toContainText("ready");
    await expect(
      page.getByTestId(`${connector.type}-card-actions`).getByRole("button", { name: `${connector.label} 연결` }),
    ).toHaveCount(0);
  }
  await expect(page.getByTestId("google-runtime-status")).toHaveCount(0);
  await expect(page.getByTestId("source-operations-panel")).toBeVisible();

  const bodyText = await page.locator("body").innerText();
  expect(bodyText).not.toContain("google-secret");
  expect(bodyText).not.toContain("refresh-token");
  expect(bodyText).not.toContain("token_ref");
  expect(bodyText).not.toContain("실제 OAuth 연동");
});

test("Gmail and Google Drive cards show connect CTAs when OAuth is configured", async ({ page }) => {
  await page.route("**/api/v1/integrations/connections", async (route) => {
    await route.fulfill({ contentType: "application/json", json: [] });
  });
  for (const connectorType of ["gmail", "drive"]) {
    await page.route(`**/api/v1/integrations/${connectorType}/oauth/install-url`, async (route) => {
      await route.fulfill({
        contentType: "application/json",
        json: {
          connector_type: connectorType,
          configured: true,
          install_url: `https://accounts.google.com/o/oauth2/v2/auth?client_id=G123&state=${connectorType}-state`,
          state: `${connectorType}-state`,
          required_scopes: [`https://www.googleapis.com/auth/${connectorType}.readonly`],
        },
      });
    });
  }

  await page.goto("/integrations");

  await expect(page.getByTestId("gmail-oauth-status").getByRole("button", { name: "Gmail 연결" })).toBeVisible();
  await expect(page.getByTestId("drive-oauth-status").getByRole("button", { name: "Google Drive 연결" })).toBeVisible();
  await expect(page.getByTestId("gmail-card-actions").locator("button")).toHaveCount(0);
  await expect(page.getByTestId("drive-card-actions").locator("button")).toHaveCount(0);
  await expect(page.getByTestId("gmail-card-actions").getByRole("button", { name: "동기화" })).toHaveCount(0);
  await expect(page.getByTestId("drive-card-actions").getByRole("button", { name: "동기화" })).toHaveCount(0);
  await expect(page.getByTestId("gmail-card-actions").getByRole("button", { name: "Gmail 연결" })).toHaveCount(0);
  await expect(page.getByTestId("drive-card-actions").getByRole("button", { name: "Google Drive 연결" })).toHaveCount(0);
});

test("Google OAuth callback route renders a safe local error without secrets", async ({ page }) => {
  await page.goto("/integrations/google/callback");

  await expect(page.getByRole("heading", { name: "Google 연결 확인" })).toBeVisible();
  await expect(page.getByText("Google 연결 정보를 확인할 수 없습니다.")).toBeVisible();

  const bodyText = await page.locator("body").innerText();
  expect(bodyText).not.toContain("google-secret");
  expect(bodyText).not.toContain("refresh-token");
  expect(bodyText).not.toContain("token_ref");
});

test("Google OAuth callback route can complete Gmail and Drive connections", async ({ page }) => {
  await page.route("**/api/v1/integrations/google/oauth/callback?**", async (route) => {
    const url = new URL(route.request().url());
    await route.fulfill({
      contentType: "application/json",
      json: {
        connector_type: url.searchParams.get("connector_type") ?? "gmail",
        workspace_id: "google-user-123",
        workspace_name: "para@example.com",
        status: "connected",
        credential_status: "available",
        masked_bot_token: "1//r...oken",
        scopes: ["https://www.googleapis.com/auth/gmail.readonly"],
      },
    });
  });

  await page.goto("/integrations/google/callback?code=temporary-code&state=signed-state");

  await expect(page.getByRole("heading", { name: "Google 연결 확인" })).toBeVisible();
  await expect(page.getByText("Google 연결 완료")).toBeVisible();
  await expect(page.getByText("para@example.com 계정이 ParaWorks에 연결되었습니다.")).toBeVisible();

  const bodyText = await page.locator("body").innerText();
  expect(bodyText).not.toContain("refresh-token");
  expect(bodyText).not.toContain("token_ref");
});
