import { expect, test, type Page, type Request } from "@playwright/test";
import { ApiError, apiGet, decodeApiErrorEnvelope } from "../src/lib/api/client";

const CAPABILITY_HEADER = "x-paraworks-rag-render-capability";
const CAPABILITY_VALUE = "rag-v2-plain-text-citations:v1";

const conversation = {
  id: 19,
  title: "기존 근거 대화",
  summary: null,
  created_at: "2026-09-12T00:00:00+00:00",
  updated_at: "2026-09-12T00:00:00+00:00",
};

const createdConversation = {
  ...conversation,
  id: 20,
  title: "새 대화 20",
  created_at: "2026-09-12T00:01:00+00:00",
  updated_at: "2026-09-12T00:01:00+00:00",
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

const emailDraftMessage = {
  ...assistantMessage,
  id: 192,
  content: "검토 후 보낼 메일 초안입니다.",
  metadata: { action_type: "email_draft", status: "pending_approval" },
};

async function captureApiGetError(body: string, status = 409) {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => new Response(body, {
    status,
    headers: { "Content-Type": "application/json" },
  })) as typeof fetch;
  try {
    await apiGet("/api/v1/test/error");
    throw new Error("expected apiGet to reject");
  } catch (error) {
    if (!(error instanceof ApiError)) throw error;
    return error;
  } finally {
    globalThis.fetch = originalFetch;
  }
}

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
      await route.fulfill({ contentType: "application/json", json: { conversations: [conversation] } });
      return;
    }
    await route.fulfill({ contentType: "application/json", json: { conversation: createdConversation } });
  });
  await page.route("**/api/v1/assistant/conversations/19/messages", async (route) => {
    assistantRequests.push(route.request());
    if (route.request().method() === "GET") {
      await route.fulfill({ contentType: "application/json", json: { conversation, messages: [emailDraftMessage] } });
      return;
    }
    throw new Error("the seeded conversation must not receive a new message");
  });
  await page.route("**/api/v1/assistant/messages/192/email/send", async (route) => {
    assistantRequests.push(route.request());
    await route.fulfill({
      contentType: "application/json",
      json: {
        message: {
          ...emailDraftMessage,
          metadata: { ...emailDraftMessage.metadata, status: "sent" },
        },
        status: "sent",
        gmail_message_id: null,
      },
    });
  });
  await page.route("**/api/v1/assistant/conversations/20/messages", async (route) => {
    assistantRequests.push(route.request());
    await route.fulfill({
      contentType: "application/json",
      json: {
        conversation: createdConversation,
        user_message: { ...userMessage, conversation_id: 20 },
        assistant_message: { ...assistantMessage, conversation_id: 20 },
      },
    });
  });

  await page.goto("/search");
  await expect(page.locator("[data-assistant-hydrated]")).toHaveAttribute("data-assistant-hydrated", "true");
  await page.getByRole("button", { name: "승인하고 보내기" }).click();
  await expect(page.getByText("전송 완료")).toBeVisible();
  const openConversationList = page.getByRole("button", { name: "대화 목록 펼치기" });
  if (await openConversationList.isVisible()) await openConversationList.click();
  await expect(page.getByRole("complementary", { name: "대화 목록" })).toHaveAttribute("data-expanded", "true");
  await page.getByRole("button", { name: "새 대화 만들기" }).click();
  await expect(page.getByRole("button", { name: "새 대화 20", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "최근 결정된 사항만 요약해줘" }).click();
  await expect(page.getByText(assistantMessage.content)).toBeVisible();

  expect(assistantRequests.map((request) => `${request.method()} ${new URL(request.url()).pathname}`)).toEqual([
    "GET /api/v1/assistant/conversations",
    "GET /api/v1/assistant/conversations/19/messages",
    "POST /api/v1/assistant/messages/192/email/send",
    "POST /api/v1/assistant/conversations",
    "POST /api/v1/assistant/conversations/20/messages",
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

test("shared decoder rejects malformed, non-exact, raw, and oversized error envelopes", async () => {
  const exact = await captureApiGetError(JSON.stringify({ detail: { code: "budget_exceeded" } }));
  expect(exact.status).toBe(409);
  expect(exact.code).toBe("budget_exceeded");
  expect(exact.message).toBe("요청이 비용 한도를 초과해 답변을 생성하지 않았습니다.");

  const malformedBodies = [
    JSON.stringify({ detail: { code: "budget_exceeded" }, unexpected: true }),
    JSON.stringify({ detail: { code: "budget_exceeded", unexpected: true } }),
    JSON.stringify([{ detail: { code: "budget_exceeded" } }]),
    "{",
    "Database password missing in production",
    `${JSON.stringify({ detail: { code: "budget_exceeded" } })}${" ".repeat(1_000_000)}`,
  ];
  for (const body of malformedBodies) {
    const error = await captureApiGetError(body, 502);
    expect(error.status).toBe(502);
    expect(error.code).toBeNull();
    expect(error.message).toBe("요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.");
    expect(error.message).not.toContain("budget_exceeded");
    expect(error.message).not.toContain("Database password");
  }
});

test("shared decoder rejects inherited and accessor-backed envelope properties", () => {
  const inheritedRoot = Object.create({ detail: { code: "budget_exceeded" } }) as object;
  const inheritedDetail = Object.create({ code: "budget_exceeded" }) as object;
  expect(decodeApiErrorEnvelope(inheritedRoot)).toBeNull();
  expect(decodeApiErrorEnvelope({ detail: inheritedDetail })).toBeNull();

  let rootAccessorRead = false;
  const accessorRoot = {};
  Object.defineProperty(accessorRoot, "detail", {
    enumerable: true,
    get() {
      rootAccessorRead = true;
      throw new Error("unsafe root accessor read");
    },
  });
  let detailAccessorRead = false;
  const accessorDetail = {};
  Object.defineProperty(accessorDetail, "code", {
    enumerable: true,
    get() {
      detailAccessorRead = true;
      throw new Error("unsafe detail accessor read");
    },
  });
  expect(decodeApiErrorEnvelope(accessorRoot)).toBeNull();
  expect(decodeApiErrorEnvelope({ detail: accessorDetail })).toBeNull();
  expect(rootAccessorRead).toBe(false);
  expect(detailAccessorRead).toBe(false);
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
