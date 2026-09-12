import { expect, test, type Page, type Request } from "@playwright/test";
import { ApiError, apiGet, decodeApiErrorEnvelope } from "../src/lib/api/client";
import { ephemeralSearchHandoff } from "../src/lib/assistant/searchHandoff";
import {
  getDeliveryStatus,
  isCurrentDelivery,
  markDelivery,
  removeOwnedOptimistic,
} from "../src/lib/assistant/reconciliation";
import { isPlainTextRagMessage, isSafeCitationUrl } from "../src/lib/rag/presentation";
import type { AssistantMessage } from "../src/lib/api/types";

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

test("Task20 pure presentation, handoff, and reconciliation contracts are fail-closed", () => {
  expect([
    "https://example.test/path",
    "http://example.test/path?q=1#section",
    "https://example.test/%E2%9C%93?q=%2F#ok",
  ].map(isSafeCitationUrl)).toEqual([true, true, true]);
  for (const unsafe of [
    "javascript:alert(1)",
    "data:text/html,unsafe",
    "//example.test/path",
    "https://user@example.test/path",
    "https://example.test/white space",
    "https://example.test/line\nbreak",
    "https://example.test/nonbreaking\u00a0space",
    "https://example.test/control\u0085character",
    "https:example.test/path",
    "https:////evil.example.test/path",
    "https:\\example.test/path",
    "https://evil.example.test/%00",
    "https://evil.example.test/%0a",
    "https://evil.example.test/%C2%A0",
    "https://evil.example.test/%C2%85",
    "https://evil.example.test/%E0%A4%A",
    "https://evil.example.test/%ZZ",
  ]) {
    expect(isSafeCitationUrl(unsafe), unsafe).toBe(false);
  }
  expect(isPlainTextRagMessage({ agent_name: "rag_orchestrator_agent", prompt_version: "rag-answer:v2" })).toBe(true);
  expect(isPlainTextRagMessage({ agent_name: "rag_orchestrator_agent", prompt_version: "future" })).toBe(true);
  expect(isPlainTextRagMessage({ agent_name: "rag_orchestrator_agent", prompt_version: "rag-answer:v1" })).toBe(false);
  expect(isPlainTextRagMessage({ agent_name: "mail_agent", prompt_version: "future" })).toBe(false);

  ephemeralSearchHandoff.consume();
  ephemeralSearchHandoff.put("  raw input  ");
  expect(ephemeralSearchHandoff.consume()).toBe("  raw input  ");
  expect(ephemeralSearchHandoff.consume()).toBeNull();
  const notifiedValues: Array<string | null> = [];
  const unsubscribe = ephemeralSearchHandoff.subscribe(() => {
    notifiedValues.push(ephemeralSearchHandoff.consume());
  });
  ephemeralSearchHandoff.put("first notification");
  ephemeralSearchHandoff.put("  second notification  ");
  unsubscribe();
  expect(notifiedValues).toEqual(["first notification", "  second notification  "]);
  expect(ephemeralSearchHandoff.consume()).toBeNull();

  const owner = { requestToken: 7, conversationId: 19, optimisticMessageId: -7 };
  const optimistic = markDelivery({ ...userMessage, id: -7 }, "unknown", owner.requestToken);
  expect(getDeliveryStatus(optimistic)).toBe("unknown");
  expect(isCurrentDelivery(owner, 7, 19)).toBe(true);
  expect(isCurrentDelivery(owner, 8, 19)).toBe(false);
  expect(isCurrentDelivery(owner, 7, 20)).toBe(false);
  expect(removeOwnedOptimistic([userMessage, optimistic], owner)).toEqual([userMessage]);
  expect(removeOwnedOptimistic([userMessage, optimistic], { ...owner, requestToken: 8 })).toEqual([userMessage, optimistic]);
});

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

function assistantRow(
  overrides: Partial<AssistantMessage> & { id: number; content: string },
): AssistantMessage {
  return {
    ...assistantMessage,
    citations: [],
    source_ids: [],
    source_links: [],
    source_snippets: [],
    permission_level: null,
    hidden_match_count: 0,
    permission_notice: null,
    agent_run_id: null,
    metadata: {},
    ...overrides,
  };
}

async function installSingleConversation(
  page: Page,
  initialMessages: ReturnType<typeof assistantRow>[],
  onMessagePost?: Parameters<Page["route"]>[1],
) {
  const otherRequests: Request[] = [];
  await fulfillShellApis(page, otherRequests);
  await page.route("**/api/v1/assistant/conversations", async (route) => {
    await route.fulfill({ contentType: "application/json", json: { conversations: [conversation] } });
  });
  await page.route("**/api/v1/assistant/conversations/19/messages", async (route, request) => {
    if (request.method() === "GET") {
      await route.fulfill({ contentType: "application/json", json: { conversation, messages: initialMessages } });
      return;
    }
    if (onMessagePost) {
      await onMessagePost(route, request);
      return;
    }
    throw new Error("unexpected Assistant message POST");
  });
}

test("V2 and unknown RAG answers stay literal while only validated citations can navigate", async ({ page }) => {
  const unsafeModelText = "# 제목\n**굵게** [보기](javascript:alert(1)) <a href=\"https://evil.test\">raw</a> https://plain.test";
  const v2 = assistantRow({
    id: 301,
    content: unsafeModelText,
    metadata: { agent_name: "rag_orchestrator_agent", prompt_version: "rag-answer:v2" },
    citations: [
      {
        source_id: "safe-source",
        source_url: "https://docs.example.test/source",
        source_type: "drive",
        permission_level: "internal",
        source_snippet: "검증된 근거",
        relevance_score: 0.98,
        matched_terms: ["근거"],
      },
      {
        source_id: "unsafe-source",
        source_url: "https://user@example.test/private",
        source_type: "drive",
        permission_level: "internal",
        source_snippet: "잘못된 URL 근거",
        relevance_score: 0.9,
        matched_terms: [],
      },
      {
        source_id: "normalized-malformed-source",
        source_url: "https:////evil.example.test/%0a",
        source_type: "drive",
        permission_level: "internal",
        source_snippet: "정규화하면 안 되는 URL 근거",
        relevance_score: 0.8,
        matched_terms: [],
      },
    ],
    source_links: ["https://legacy.example.test/must-not-link", "javascript:alert(2)"],
    source_snippets: ["검증된 근거", "추가 발췌"],
    hidden_match_count: 2,
    permission_notice: "Some sources may be hidden by permissions.",
  });
  const unknown = assistantRow({
    id: 302,
    content: "## 알 수 없는 버전\n[링크](https://unknown.test)",
    metadata: { agent_name: "rag_orchestrator_agent", prompt_version: "rag-answer:future" },
  });
  const legacy = assistantRow({
    id: 303,
    content: "**레거시 강조**\n- 기존 목록",
    metadata: { agent_name: "rag_orchestrator_agent", prompt_version: "rag-answer:v1" },
  });
  await installSingleConversation(page, [v2, unknown, legacy]);

  await page.goto("/search");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  const v2Article = page.locator("article").filter({ hasText: "javascript:alert(1)" });
  const literal = v2Article.getByTestId("rag-plain-text-answer");
  await expect(literal).toHaveText(unsafeModelText);
  await expect(literal).toHaveCSS("white-space", "pre-wrap");
  await expect(literal).toHaveCSS("overflow-wrap", "anywhere");
  await expect(v2Article.locator("h3, strong, code, a")).toHaveCount(0);
  await expect(v2Article.getByText("일부 근거는 권한에 따라 숨겨졌습니다.")).toBeVisible();
  await expect(page.locator("body")).not.toContainText("Some sources may be hidden by permissions.");

  const unknownArticle = page.locator("article").filter({ hasText: "알 수 없는 버전" });
  await expect(unknownArticle.getByTestId("rag-plain-text-answer")).toContainText("[링크](https://unknown.test)");
  await expect(unknownArticle.locator("h3, a")).toHaveCount(0);

  const legacyArticle = page.locator("article").filter({ hasText: "레거시 강조" });
  await expect(legacyArticle.locator("strong")).toHaveText("레거시 강조");
  await expect(legacyArticle.locator("li")).toHaveText("기존 목록");

  await v2Article.getByRole("button", { name: /근거와 출처/ }).click();
  const safeLink = v2Article.getByRole("link", { name: /근거 1/ });
  await expect(safeLink).toHaveAttribute("href", "https://docs.example.test/source");
  await expect(safeLink).toHaveAttribute("rel", "noopener noreferrer");
  await expect(v2Article.getByText("unsafe-source", { exact: true })).toBeVisible();
  await expect(v2Article.getByText("unsafe-source", { exact: true }).locator("xpath=ancestor::a")).toHaveCount(0);
  await expect(v2Article.getByText("normalized-malformed-source", { exact: true })).toBeVisible();
  await expect(v2Article.getByText("normalized-malformed-source", { exact: true }).locator("xpath=ancestor::a")).toHaveCount(0);
  await expect(v2Article).not.toContainText("https:////evil.example.test/%0a");
  await expect(v2Article.locator("a")).toHaveCount(1);
  await expect(v2Article.locator('a[href*="legacy.example"]')).toHaveCount(0);
});

test("shell hands raw search input to /search once without URL or automatic POST", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name.includes("mobile"), "mobile shell has no global search input");

  const rawInput = "  민감할 수 있는 원문  ";
  let messagePostCount = 0;
  await installSingleConversation(page, [], async (route) => {
    messagePostCount += 1;
    await route.fulfill({ status: 500, body: "must not send" });
  });

  const authResponse = page.waitForResponse("**/api/v1/auth/me");
  await page.goto("/dashboard");
  await authResponse;
  const shellSearch = page.getByRole("textbox", { name: "회사 메모리 검색" });
  await shellSearch.fill(rawInput);
  await shellSearch.press("Enter");

  await expect(page).toHaveURL(/\/search$/);
  await expect(page.getByRole("textbox", { name: "AI 비서에게 질문" })).toHaveValue(rawInput);
  expect(messagePostCount).toBe(0);
  const browserStorage = await page.evaluate((secret) => ({
    href: location.href,
    history: JSON.stringify(history.state),
    local: JSON.stringify(localStorage),
    session: JSON.stringify(sessionStorage),
    hasSecret: [location.href, JSON.stringify(history.state), JSON.stringify(localStorage), JSON.stringify(sessionStorage)]
      .some((value) => value.includes(secret)),
  }), rawInput.trim());
  expect(browserStorage.hasSecret, JSON.stringify(browserStorage)).toBe(false);

  await page.reload();
  await expect(page.getByRole("textbox", { name: "AI 비서에게 질문" })).toHaveValue("");
  expect(messagePostCount).toBe(0);
});

test("same-route shell search is consumed immediately and never reappears after client navigation", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name.includes("mobile"), "mobile shell has no global search input");

  let messagePostCount = 0;
  await installSingleConversation(page, [], async (route) => {
    messagePostCount += 1;
    await route.fulfill({ status: 500, body: "must not send" });
  });
  const authResponse = page.waitForResponse("**/api/v1/auth/me");
  await page.goto("/search");
  await authResponse;
  const assistantInput = page.getByRole("textbox", { name: "AI 비서에게 질문" });
  const shellSearch = page.getByRole("textbox", { name: "회사 메모리 검색" });
  await page.evaluate(() => {
    const taskWindow = window as unknown as Window & { __task20HandoffEvents: number };
    taskWindow.__task20HandoffEvents = 0;
    window.addEventListener("paraworks:ephemeral-search-handoff-ready", () => {
      taskWindow.__task20HandoffEvents += 1;
    });
  });

  await shellSearch.fill("  same-route-secret  ");
  await shellSearch.press("Enter");
  await expect(page).toHaveURL(/\/search$/);
  expect(await page.evaluate(() => (
    window as unknown as Window & { __task20HandoffEvents: number }
  ).__task20HandoffEvents)).toBe(1);
  await expect(assistantInput).toHaveValue("  same-route-secret  ");

  await shellSearch.fill("  last-write-only  ");
  await shellSearch.press("Enter");
  await expect(assistantInput).toHaveValue("  last-write-only  ");

  await page.getByRole("link", { name: "대시보드", exact: true }).first().click();
  await expect(page).toHaveURL(/\/dashboard$/);
  await page.getByRole("link", { name: "AI 비서", exact: true }).first().click();
  await expect(page).toHaveURL(/\/search$/);
  await expect(page.getByRole("textbox", { name: "AI 비서에게 질문" })).toHaveValue("");
  expect(messagePostCount).toBe(0);
});

test("rapid other-route handoffs expose only the last raw value and leave no stale re-entry", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name.includes("mobile"), "mobile shell has no global search input");

  await installSingleConversation(page, []);
  const authResponse = page.waitForResponse("**/api/v1/auth/me");
  await page.goto("/dashboard");
  await authResponse;
  const shellSearch = page.getByRole("textbox", { name: "회사 메모리 검색" });
  await shellSearch.fill("  first-route-value  ");
  await shellSearch.evaluate((input) => {
    const form = input.closest("form");
    if (!(input instanceof HTMLInputElement) || !(form instanceof HTMLFormElement)) {
      throw new Error("expected the existing shell search form");
    }
    form.requestSubmit();
    input.value = "  last-route-value  ";
    form.requestSubmit();
  });

  await expect(page).toHaveURL(/\/search$/);
  await expect(page.getByRole("textbox", { name: "AI 비서에게 질문" })).toHaveValue("  last-route-value  ");
  await page.getByRole("link", { name: "대시보드", exact: true }).first().click();
  await page.getByRole("link", { name: "AI 비서", exact: true }).first().click();
  await expect(page.getByRole("textbox", { name: "AI 비서에게 질문" })).toHaveValue("");
});

test("legacy inbound q is discarded and never copied or auto-sent", async ({ page }) => {
  let messagePostCount = 0;
  await installSingleConversation(page, [], async (route) => {
    messagePostCount += 1;
    await route.fulfill({ status: 500, body: "must not send" });
  });

  await page.goto("/search?q=credential-like-value");
  await expect(page).toHaveURL(/\/search$/);
  await expect(page.getByRole("textbox", { name: "AI 비서에게 질문" })).toHaveValue("");
  expect(messagePostCount).toBe(0);
});

for (const refusal of [
  { status: 403, code: "permission_denied", copy: "이 요청을 처리할 권한이 없습니다." },
  { status: 404, code: null, copy: "대화를 찾을 수 없습니다. 대화 목록을 새로고침해 주세요." },
  { status: 422, code: "input_safety_blocked", copy: "민감한 정보로 보이는 내용이 포함되어 요청을 전송하지 않았습니다." },
]) {
  test(`${refusal.status} removes only the request-owned optimistic row with safe Korean copy`, async ({ page }) => {
    await installSingleConversation(page, [], async (route) => {
      await route.fulfill({
        status: refusal.status,
        contentType: "application/json",
        json: refusal.code ? { detail: { code: refusal.code } } : { detail: "owner hidden" },
      });
    });
    await page.goto("/search");
    const input = page.getByRole("textbox", { name: "AI 비서에게 질문" });
    await input.fill("제거할 낙관적 메시지");
    await input.press("Enter");
    await expect(page.getByText(refusal.copy)).toBeVisible();
    await expect(page.getByText("제거할 낙관적 메시지", { exact: true })).toHaveCount(0);
    await expect(page.locator("body")).not.toContainText(refusal.code ?? "owner hidden");
  });
}

test("500 leaves one unknown row when guarded GET fails and retry is GET-only", async ({ page }) => {
  let getCount = 0;
  let postCount = 0;
  const persistedUser = assistantRow({ id: 401, role: "user", content: "상태가 불확실한 요청" });
  const persistedAssistant = assistantRow({
    id: 402,
    content: "권한 내에서 확인 가능한 근거를 찾지 못했습니다.",
    metadata: { agent_name: "rag_orchestrator_agent", prompt_version: "rag-answer:v2" },
  });
  const otherRequests: Request[] = [];
  await fulfillShellApis(page, otherRequests);
  await page.route("**/api/v1/assistant/conversations", (route) => route.fulfill({ json: { conversations: [conversation] } }));
  await page.route("**/api/v1/assistant/conversations/19/messages", async (route, request) => {
    if (request.method() === "POST") {
      postCount += 1;
      await route.fulfill({ status: 500, contentType: "application/json", json: { detail: { code: "persistence_failed" } } });
      return;
    }
    getCount += 1;
    if (getCount === 1) {
      await route.fulfill({ json: { conversation, messages: [] } });
    } else if (getCount === 2) {
      await route.fulfill({ status: 503, body: "private GET failure" });
    } else {
      await route.fulfill({ json: { conversation, messages: [persistedUser, persistedAssistant] } });
    }
  });

  await page.goto("/search");
  const input = page.getByRole("textbox", { name: "AI 비서에게 질문" });
  await input.fill(persistedUser.content);
  await input.press("Enter");
  await expect(page.getByText("요청 처리 상태를 확인하지 못했습니다. 잠시 후 다시 시도해 주세요.")).toBeVisible();
  await expect(page.getByText(persistedUser.content, { exact: true })).toHaveCount(1);
  await expect(page.getByLabel("AI 비서 입력").getByText("전송 상태를 확인하지 못했습니다.")).toBeVisible();
  await expect(input).toBeDisabled();
  await expect(page.getByRole("button", { name: "새 대화 만들기", includeHidden: true })).toBeDisabled();

  await page.getByRole("button", { name: "상태 다시 확인" }).click();
  await expect(page.getByText(persistedUser.content, { exact: true })).toHaveCount(1);
  await expect(page.getByText(persistedAssistant.content)).toBeVisible();
  await expect(page.getByRole("button", { name: "상태 다시 확인" })).toHaveCount(0);
  await expect(input).toBeEnabled();
  expect(postCount).toBe(1);
  expect(getCount).toBe(3);
});

test("an exact client-upgrade reconciliation GET enters reload-only without a GET retry", async ({ page }) => {
  let getCount = 0;
  let postCount = 0;
  const otherRequests: Request[] = [];
  await fulfillShellApis(page, otherRequests);
  await page.route("**/api/v1/assistant/conversations", (route) => route.fulfill({
    json: { conversations: [conversation] },
  }));
  await page.route("**/api/v1/assistant/conversations/19/messages", async (route, request) => {
    if (request.method() === "POST") {
      postCount += 1;
      await route.fulfill({ status: 500, contentType: "application/json", json: { detail: { code: "persistence_failed" } } });
      return;
    }
    getCount += 1;
    if (getCount === 1) {
      await route.fulfill({ json: { conversation, messages: [] } });
      return;
    }
    await route.fulfill({
      status: 409,
      contentType: "application/json",
      json: { detail: { code: "client_upgrade_required" } },
    });
  });

  await page.goto("/search");
  const input = page.getByRole("textbox", { name: "AI 비서에게 질문" });
  await input.fill("조정 중 업그레이드 요청");
  await input.press("Enter");
  await expect(page.getByText("새 버전이 필요합니다. 페이지를 새로고침해 주세요.")).toBeVisible();
  await expect(page.getByRole("button", { name: "페이지 새로고침" })).toBeVisible();
  await expect(page.getByRole("button", { name: "상태 다시 확인" })).toHaveCount(0);
  await expect(page.getByText("조정 중 업그레이드 요청", { exact: true })).toHaveCount(1);
  await expect(input).toBeDisabled();
  expect(postCount).toBe(1);
  expect(getCount).toBe(2);
});

test("502 generation failure performs guarded GET and never renders raw failure text", async ({ page }) => {
  let postCount = 0;
  await installSingleConversation(page, [], async (route) => {
    postCount += 1;
    await route.fulfill({
      status: 502,
      contentType: "text/plain",
      body: "private provider exception",
    });
  });
  await page.goto("/search");
  const input = page.getByRole("textbox", { name: "AI 비서에게 질문" });
  await input.fill("생성 실패 요청");
  await input.press("Enter");
  await expect(page.getByText("답변을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요.")).toBeVisible();
  await expect(page.getByText("생성 실패 요청", { exact: true })).toHaveCount(0);
  await expect(page.locator("body")).not.toContainText("private provider exception");
  await expect(input).toBeEnabled();
  expect(postCount).toBe(1);
});

test("budget conflict keeps its safe copy when authoritative reconciliation is unavailable", async ({ page }) => {
  let getCount = 0;
  const otherRequests: Request[] = [];
  await fulfillShellApis(page, otherRequests);
  await page.route("**/api/v1/assistant/conversations", (route) => route.fulfill({ json: { conversations: [conversation] } }));
  await page.route("**/api/v1/assistant/conversations/19/messages", async (route, request) => {
    if (request.method() === "POST") {
      await route.fulfill({ status: 409, contentType: "application/json", json: { detail: { code: "budget_exceeded" } } });
      return;
    }
    getCount += 1;
    if (getCount === 1) {
      await route.fulfill({ json: { conversation, messages: [] } });
    } else {
      await route.fulfill({ status: 503, body: "private reconciliation failure" });
    }
  });

  await page.goto("/search");
  const input = page.getByRole("textbox", { name: "AI 비서에게 질문" });
  await input.fill("비용 한도 미확인 요청");
  await input.press("Enter");
  await expect(page.getByText("요청이 비용 한도를 초과해 답변을 생성하지 않았습니다.")).toBeVisible();
  await expect(page.getByLabel("AI 비서 입력").getByText("전송 상태를 확인하지 못했습니다.")).toBeVisible();
  await expect(page.locator("body")).not.toContainText("budget_exceeded");
});

test("budget conflict reconciles committed rows and client upgrade permits reload only", async ({ page }) => {
  let postCount = 0;
  let getCount = 0;
  const budgetUser = assistantRow({ id: 501, role: "user", content: "비용 한도 질문" });
  const budgetAssistant = assistantRow({
    id: 502,
    content: "요청이 비용 한도를 초과해 답변을 생성하지 않았습니다.",
    metadata: { agent_name: "rag_orchestrator_agent", prompt_version: "rag-answer:v2", status: "failed" },
  });
  const otherRequests: Request[] = [];
  await fulfillShellApis(page, otherRequests);
  await page.route("**/api/v1/assistant/conversations", (route) => route.fulfill({ json: { conversations: [conversation] } }));
  await page.route("**/api/v1/assistant/conversations/19/messages", async (route, request) => {
    if (request.method() === "GET") {
      getCount += 1;
      await route.fulfill({ json: { conversation, messages: getCount === 1 ? [] : [budgetUser, budgetAssistant] } });
      return;
    }
    postCount += 1;
    await route.fulfill({ status: 409, contentType: "application/json", json: { detail: { code: "budget_exceeded" } } });
  });

  await page.goto("/search");
  const input = page.getByRole("textbox", { name: "AI 비서에게 질문" });
  await input.fill(budgetUser.content);
  await input.press("Enter");
  await expect(page.getByText(budgetUser.content, { exact: true })).toHaveCount(1);
  await expect(page.getByText(budgetAssistant.content)).toBeVisible();
  expect(postCount).toBe(1);
  expect(getCount).toBe(2);

  await page.unroute("**/api/v1/assistant/conversations/19/messages");
  await page.route("**/api/v1/assistant/conversations/19/messages", async (route, request) => {
    if (request.method() === "GET") {
      await route.fulfill({ json: { conversation, messages: [budgetUser, budgetAssistant] } });
      return;
    }
    postCount += 1;
    await route.fulfill({ status: 409, contentType: "application/json", json: { detail: { code: "client_upgrade_required" } } });
  });
  await input.fill("새 클라이언트가 필요한 질문");
  await input.press("Enter");
  await expect(page.getByText("새 버전이 필요합니다. 페이지를 새로고침해 주세요.")).toBeVisible();
  await expect(page.getByRole("button", { name: "페이지 새로고침" })).toBeVisible();
  await expect(page.getByRole("button", { name: "상태 다시 확인" })).toHaveCount(0);
  await expect(input).toBeDisabled();
  await expect(page.getByRole("button", { name: "기존 근거 대화", exact: true, includeHidden: true })).toBeDisabled();
  expect(postCount).toBe(2);
});

test("client-upgrade on authoritative GET blocks every POST and offers hard reload only", async ({ page }) => {
  let postCount = 0;
  const otherRequests: Request[] = [];
  await fulfillShellApis(page, otherRequests);
  await page.route("**/api/v1/assistant/conversations", async (route, request) => {
    if (request.method() === "POST") postCount += 1;
    await route.fulfill({
      status: 409,
      contentType: "application/json",
      json: { detail: { code: "client_upgrade_required" } },
    });
  });

  await page.goto("/search");
  await expect(page.getByText("새 버전이 필요합니다. 페이지를 새로고침해 주세요.")).toBeVisible();
  await expect(page.getByRole("button", { name: "페이지 새로고침" })).toBeVisible();
  await expect(page.getByRole("button", { name: "상태 다시 확인" })).toHaveCount(0);
  await expect(page.getByRole("textbox", { name: "AI 비서에게 질문" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "새 대화 만들기", includeHidden: true })).toBeDisabled();
  expect(postCount).toBe(0);
});

test("a delayed stale POST client-upgrade latches globally without replacing the selected conversation", async ({ page }) => {
  const secondConversation = {
    ...createdConversation,
    id: 21,
    title: "업그레이드 중 선택한 대화",
    created_at: "2026-09-11T00:00:00+00:00",
    updated_at: "2026-09-11T00:00:00+00:00",
  };
  const secondEmailDraft = {
    ...emailDraftMessage,
    id: 621,
    conversation_id: 21,
    content: "새 대화의 보존할 메일 초안",
  };
  let releasePost!: () => void;
  let signalPostStarted!: () => void;
  const postGate = new Promise<void>((resolve) => { releasePost = resolve; });
  const postStarted = new Promise<void>((resolve) => { signalPostStarted = resolve; });
  let postCount = 0;
  const otherRequests: Request[] = [];
  await fulfillShellApis(page, otherRequests);
  await page.route("**/api/v1/assistant/conversations", (route) => route.fulfill({
    json: { conversations: [conversation, secondConversation] },
  }));
  await page.route("**/api/v1/assistant/conversations/19/messages", async (route, request) => {
    if (request.method() === "GET") {
      await route.fulfill({ json: { conversation, messages: [] } });
      return;
    }
    postCount += 1;
    signalPostStarted();
    await postGate;
    await route.fulfill({
      status: 409,
      contentType: "application/json",
      json: { detail: { code: "client_upgrade_required" } },
    });
  });
  await page.route("**/api/v1/assistant/conversations/21/messages", (route) => route.fulfill({
    json: { conversation: secondConversation, messages: [secondEmailDraft] },
  }));

  await page.goto("/search");
  const input = page.getByRole("textbox", { name: "AI 비서에게 질문" });
  await input.fill("늦은 업그레이드 요청");
  await input.press("Enter");
  await postStarted;
  const openConversationList = page.getByRole("button", { name: "대화 목록 펼치기" });
  if (await openConversationList.isVisible()) await openConversationList.click();
  await page.getByRole("button", { name: secondConversation.title, exact: true }).click();
  await expect(page.getByText(secondEmailDraft.content)).toBeVisible();
  releasePost();

  await expect(page.getByText("새 버전이 필요합니다. 페이지를 새로고침해 주세요.")).toBeVisible();
  await expect(page.getByRole("button", { name: "페이지 새로고침" })).toBeVisible();
  await expect(page.getByText(secondEmailDraft.content)).toBeVisible();
  await expect(page.getByText("늦은 업그레이드 요청", { exact: true })).toHaveCount(0);
  await expect(input).toBeDisabled();
  await expect(page.getByRole("button", { name: "승인하고 보내기" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "새 대화 만들기" })).toBeDisabled();
  await expect(page.getByRole("button", { name: secondConversation.title, exact: true })).toBeDisabled();
  expect(postCount).toBe(1);
});

test("a stale reconciliation GET cannot overwrite a newly selected conversation", async ({ page }) => {
  const secondConversation = {
    ...createdConversation,
    id: 21,
    title: "새 대화 선택",
    created_at: "2026-09-11T00:00:00+00:00",
    updated_at: "2026-09-11T00:00:00+00:00",
  };
  const secondMessage = assistantRow({ id: 610, conversation_id: 21, content: "새 대화의 현재 내용" });
  let releaseStale!: () => void;
  const staleGate = new Promise<void>((resolve) => { releaseStale = resolve; });
  let firstGetCount = 0;
  const otherRequests: Request[] = [];
  await fulfillShellApis(page, otherRequests);
  await page.route("**/api/v1/assistant/conversations", (route) => route.fulfill({
    json: { conversations: [conversation, secondConversation] },
  }));
  await page.route("**/api/v1/assistant/conversations/19/messages", async (route, request) => {
    if (request.method() === "POST") {
      await route.fulfill({ status: 500, contentType: "application/json", json: { detail: { code: "persistence_failed" } } });
      return;
    }
    firstGetCount += 1;
    if (firstGetCount === 1) {
      await route.fulfill({ json: { conversation, messages: [] } });
      return;
    }
    await staleGate;
    await route.fulfill({
      json: { conversation, messages: [assistantRow({ id: 611, content: "이전 대화의 늦은 응답" })] },
    });
  });
  await page.route("**/api/v1/assistant/conversations/21/messages", (route) => route.fulfill({
    json: { conversation: secondConversation, messages: [secondMessage] },
  }));

  await page.goto("/search");
  const input = page.getByRole("textbox", { name: "AI 비서에게 질문" });
  await input.fill("늦게 끝나는 요청");
  await input.press("Enter");
  const openConversationList = page.getByRole("button", { name: "대화 목록 펼치기" });
  if (await openConversationList.isVisible()) await openConversationList.click();
  await page.getByRole("button", { name: "새 대화 선택", exact: true }).click();
  await expect(page.getByText(secondMessage.content)).toBeVisible();
  releaseStale();
  await page.waitForTimeout(100);
  await expect(page.getByText(secondMessage.content)).toBeVisible();
  await expect(page.getByText("이전 대화의 늦은 응답")).toHaveCount(0);
});

test("late Assistant success after Search unmount creates no typing interval or unrelated UI", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name.includes("mobile"), "desktop shell navigation drives this unmount probe");

  let releasePost!: () => void;
  let signalPostStarted!: () => void;
  const postGate = new Promise<void>((resolve) => { releasePost = resolve; });
  const postStarted = new Promise<void>((resolve) => { signalPostStarted = resolve; });
  await installSingleConversation(page, [], async (route) => {
    signalPostStarted();
    await postGate;
    await route.fulfill({
      contentType: "application/json",
      json: {
        conversation,
        user_message: assistantRow({ id: 701, role: "user", content: "언마운트 지연 요청" }),
        assistant_message: assistantRow({ id: 702, content: "언마운트 뒤 나타나면 안 되는 응답" }),
      },
    });
  });
  await page.goto("/search");
  await page.evaluate(() => {
    const taskWindow = window as unknown as Window & { __task20TypingIntervals: number };
    const originalSetInterval = window.setInterval.bind(window);
    taskWindow.__task20TypingIntervals = 0;
    window.setInterval = ((handler: TimerHandler, timeout?: number, ...args: unknown[]) => {
      if (timeout === 18) taskWindow.__task20TypingIntervals += 1;
      return originalSetInterval(handler, timeout, ...args);
    }) as typeof window.setInterval;
  });
  const input = page.getByRole("textbox", { name: "AI 비서에게 질문" });
  await input.fill("언마운트 지연 요청");
  await input.press("Enter");
  await postStarted;
  await page.getByRole("link", { name: "대시보드", exact: true }).first().click();
  await expect(page).toHaveURL(/\/dashboard$/);
  releasePost();
  await page.waitForTimeout(100);

  expect(await page.evaluate(() => (
    window as unknown as Window & { __task20TypingIntervals: number }
  ).__task20TypingIntervals)).toBe(0);
  await expect(page.getByText("언마운트 뒤 나타나면 안 되는 응답")).toHaveCount(0);
});

test("late Assistant failure after Search unmount cannot latch a later Search mount", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name.includes("mobile"), "desktop shell navigation drives this unmount probe");

  let releasePost!: () => void;
  let signalPostStarted!: () => void;
  const postGate = new Promise<void>((resolve) => { releasePost = resolve; });
  const postStarted = new Promise<void>((resolve) => { signalPostStarted = resolve; });
  await installSingleConversation(page, [], async (route) => {
    signalPostStarted();
    await postGate;
    await route.fulfill({
      status: 409,
      contentType: "application/json",
      json: { detail: { code: "client_upgrade_required" } },
    });
  });
  await page.goto("/search");
  const input = page.getByRole("textbox", { name: "AI 비서에게 질문" });
  await input.fill("언마운트 지연 실패");
  await input.press("Enter");
  await postStarted;
  await page.getByRole("link", { name: "대시보드", exact: true }).first().click();
  await expect(page).toHaveURL(/\/dashboard$/);
  releasePost();
  await page.waitForTimeout(100);
  await page.getByRole("link", { name: "AI 비서", exact: true }).first().click();
  await expect(page).toHaveURL(/\/search$/);

  await expect(page.getByText("새 버전이 필요합니다. 페이지를 새로고침해 주세요.")).toHaveCount(0);
  await expect(page.getByRole("textbox", { name: "AI 비서에게 질문" })).toBeEnabled();
});
