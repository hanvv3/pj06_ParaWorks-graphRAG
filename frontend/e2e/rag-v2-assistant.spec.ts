import { expect, test, type Page, type Request } from "@playwright/test";

const CAPABILITY_HEADER = "x-paraworks-rag-render-capability";
const CAPABILITY_VALUE = "rag-v2-plain-text-citations:v1";

const conversation = {
  id: 19,
  title: "새 대화",
  summary: null,
  created_at: "2026-09-12T00:00:00+00:00",
  updated_at: "2026-09-12T00:00:00+00:00",
};

const userMessage = {
  id: 190,
  conversation_id: 19,
  role: "user",
  content: "최근 결정을 알려줘",
  citations: [],
  source_ids: [],
  source_links: [],
  source_snippets: [],
  permission_level: null,
  hidden_match_count: 0,
  permission_notice: null,
  agent_run_id: null,
  metadata: {},
  created_at: "2026-09-12T00:00:01+00:00",
};

const assistantMessage = {
  ...userMessage,
  id: 191,
  role: "assistant",
  content: "권한 내에서 확인 가능한 근거를 찾지 못했습니다.",
  permission_level: "internal",
  agent_run_id: 91,
  created_at: "2026-09-12T00:00:02+00:00",
};

async function fulfillShellApis(page: Page, observedOtherRequests: Request[]) {
  await page.route("**/api/v1/auth/me", async (route) => {
    observedOtherRequests.push(route.request());
    await route.fulfill({
      contentType: "application/json",
      json: {
        user: {
          id: "employee-mina",
          email: "mina@paraworks.com",
          role: "reviewer",
          permission_levels: ["public", "internal"],
          name: "Kim Mina",
          title: "Product Manager",
          department: "Product",
        },
      },
    });
  });
  await page.route("**/api/v1/dashboard", async (route) => {
    observedOtherRequests.push(route.request());
    await route.fulfill({ contentType: "application/json", json: {} });
  });
  await page.route("**/api/v1/notifications", async (route) => {
    observedOtherRequests.push(route.request());
    await route.fulfill({
      contentType: "application/json",
      json: { counts: { total: 0, review: 0, agent_runs: 0 }, notifications: [] },
    });
  });
}

async function capabilityValues(request: Request) {
  return (await request.headersArray())
    .filter(({ name }) => name.toLowerCase() === CAPABILITY_HEADER)
    .map(({ value }) => value);
}

test("Assistant GET and POST declare one exact render capability without changing shared API headers", async ({
  context,
  page,
}) => {
  await context.addCookies([
    { name: "paraworks_csrf", value: "csrf-test-token", url: "http://localhost:3000" },
  ]);
  const assistantRequests: Request[] = [];
  const otherRequests: Request[] = [];
  await fulfillShellApis(page, otherRequests);

  await page.route("**/api/v1/assistant/conversations", async (route) => {
    assistantRequests.push(route.request());
    if (route.request().method() === "GET") {
      await route.fulfill({ contentType: "application/json", json: { conversations: [] } });
      return;
    }
    await route.fulfill({ contentType: "application/json", json: { conversation } });
  });
  await page.route("**/api/v1/assistant/conversations/19/messages", async (route) => {
    assistantRequests.push(route.request());
    if (route.request().method() === "GET") {
      await route.fulfill({ contentType: "application/json", json: { conversation, messages: [] } });
      return;
    }
    await route.fulfill({
      contentType: "application/json",
      json: { conversation, user_message: userMessage, assistant_message: assistantMessage },
    });
  });

  await page.goto("/search");
  await expect(page.locator("[data-assistant-hydrated]")).toHaveAttribute("data-assistant-hydrated", "true");
  await page.getByRole("button", { name: "최근 결정된 사항만 요약해줘" }).click();
  await expect(page.getByText(assistantMessage.content)).toBeVisible();

  expect(assistantRequests.map((request) => `${request.method()} ${new URL(request.url()).pathname}`)).toEqual([
    "GET /api/v1/assistant/conversations",
    "POST /api/v1/assistant/conversations",
    "POST /api/v1/assistant/conversations/19/messages",
  ]);
  for (const request of assistantRequests) {
    expect(await capabilityValues(request)).toEqual([CAPABILITY_VALUE]);
    expect(request.headers()["x-demo-user"]).toBe("hanvv-employee");
  }
  const assistantPosts = assistantRequests.filter((request) => request.method() === "POST");
  for (const request of assistantPosts) {
    expect(request.headers()["content-type"]).toContain("application/json");
    expect(request.headers()["x-csrf-token"]).toBe("csrf-test-token");
  }
  expect(otherRequests.length).toBeGreaterThan(0);
  for (const request of otherRequests) {
    expect(await capabilityValues(request)).toEqual([]);
  }
});

for (const errorCase of [
  {
    name: "an unknown structured code",
    status: 502,
    body: JSON.stringify({ detail: { code: "private_backend_secret" } }),
    forbidden: "private_backend_secret",
  },
  {
    name: "an inherited object property code",
    status: 502,
    body: JSON.stringify({ detail: { code: "toString" } }),
    forbidden: "toString",
  },
  {
    name: "raw English response text",
    status: 500,
    body: "Database password missing in production",
    forbidden: "Database password missing in production",
  },
]) {
  test(`Assistant errors replace ${errorCase.name} with bounded Korean copy`, async ({ page }) => {
    const otherRequests: Request[] = [];
    await fulfillShellApis(page, otherRequests);
    await page.route("**/api/v1/assistant/conversations", async (route) => {
      await route.fulfill({
        status: errorCase.status,
        contentType: "application/json",
        body: errorCase.body,
      });
    });

    await page.goto("/search");
    const alert = page.locator(".border-red-200");
    await expect(alert).toContainText("요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.");
    await expect(alert).not.toContainText(errorCase.forbidden);
    await expect(page.locator("body")).not.toContainText(errorCase.forbidden);
  });
}

test("Assistant errors decode only an exact allowlisted code to stable Korean copy", async ({ page }) => {
  const otherRequests: Request[] = [];
  await fulfillShellApis(page, otherRequests);
  await page.route("**/api/v1/assistant/conversations", async (route) => {
    await route.fulfill({
      status: 422,
      contentType: "application/json",
      json: { detail: { code: "input_safety_blocked" } },
    });
  });

  await page.goto("/search");
  const alert = page.locator(".border-red-200");
  await expect(alert).toContainText("민감한 정보로 보이는 내용이 포함되어 요청을 전송하지 않았습니다.");
  await expect(page.locator("body")).not.toContainText("input_safety_blocked");
});
