import { expect, test } from "@playwright/test";

test("Timeline groups approved project items by date", async ({ page }) => {
  await page.route("**/api/v1/auth/me", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: {
        user: {
          id: "demo-admin",
          email: "admin@paraworks.com",
          role: "admin",
          permission_levels: ["public", "internal"],
          name: "Admin",
          title: "Admin",
          department: "Platform",
        },
      },
    });
  });
  await page.route("**/api/v1/notifications", async (route) => {
    await route.fulfill({ contentType: "application/json", json: { unread_count: 0, notifications: [] } });
  });
  await page.route("**/api/v1/projects", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: {
        project_count: 1,
        hidden_project_count: 0,
        hidden_evidence_count: 0,
        projects: [
          {
            project_key: "project-alpha",
            name: "Project Alpha",
            summary: "Redis work",
            source_types: ["slack"],
            evidence_count: 0,
            permission_level: "internal",
            latest_timestamp: "2026-05-15T02:00:00Z",
            pending_review_count: 0,
            evidence: [],
            activity_items: [],
            timeline_items: [
              {
                id: "timeline_event:1",
                item_type: "timeline_event",
                title: "오전 점검",
                summary: "Redis 점검",
                source_links: ["https://slack.example/1"],
                source_snippets: ["점검"],
                confidence_score: 0.9,
                permission_level: "internal",
                review_status: "approved",
                created_at: "2026-05-15T09:00:00Z",
                occurred_at: "2026-05-14T01:00:00+09:00",
                evidence_reason: "승인된 항목",
                project_key: "project-alpha",
                resolution_source: "auto_policy",
              },
              {
                id: "timeline_event:2",
                item_type: "timeline_event",
                title: "오후 배포",
                summary: "배포 완료",
                source_links: ["https://slack.example/2"],
                source_snippets: ["배포"],
                confidence_score: 0.9,
                permission_level: "internal",
                review_status: "approved",
                created_at: "2026-05-15T10:00:00Z",
                occurred_at: "2026-05-14T06:00:00+09:00",
                evidence_reason: "승인된 항목",
                project_key: "project-alpha",
              },
              {
                id: "timeline_event:4",
                item_type: "timeline_event",
                title: "QA 승인",
                summary: "QA 승인 완료",
                source_links: ["https://slack.example/4"],
                source_snippets: ["QA 승인"],
                confidence_score: 0.9,
                permission_level: "internal",
                review_status: "approved",
                created_at: "2026-05-15T12:00:00Z",
                occurred_at: "2026-05-14T07:00:00+09:00",
                evidence_reason: "승인된 항목",
                project_key: "project-alpha",
              },
              {
                id: "timeline_event:5",
                item_type: "timeline_event",
                title: "릴리즈 노트 정리",
                summary: "릴리즈 노트 완료",
                source_links: ["https://slack.example/5"],
                source_snippets: ["릴리즈 노트"],
                confidence_score: 0.9,
                permission_level: "internal",
                review_status: "approved",
                created_at: "2026-05-15T13:00:00Z",
                occurred_at: "2026-05-14T08:00:00+09:00",
                evidence_reason: "승인된 항목",
                project_key: "project-alpha",
              },
              {
                id: "timeline_event:3",
                item_type: "timeline_event",
                title: "전날 회의",
                summary: "회의 완료",
                source_links: ["https://slack.example/3"],
                source_snippets: ["회의"],
                confidence_score: 0.9,
                permission_level: "internal",
                review_status: "approved",
                created_at: "2026-05-15T11:00:00Z",
                occurred_at: "2026-05-13T06:00:00+09:00",
                evidence_reason: "승인된 항목",
                project_key: "project-alpha",
              },
            ],
          },
        ],
      },
    });
  });
  await page.addInitScript(() => window.localStorage.setItem("paraworks-demo-user", "demo-admin"));

  await page.goto("/timeline");

  await expect(page.getByLabel("소스 필터")).toContainText("소스 전체");
  await expect(page.getByLabel("소스 필터")).toContainText("Slack");
  await expect(page.getByLabel("소스 필터")).toContainText("Gmail");
  await expect(page.getByLabel("소스 필터")).toContainText("Drive");
  await expect(page.getByLabel("소스 필터")).toContainText("Calendar");
  await expect(page.getByLabel("소스 필터")).not.toContainText("Source");
  await expect(page.getByLabel("상태 필터")).toContainText("상태 전체");
  await expect(page.getByLabel("상태 필터")).toContainText("승인");
  await expect(page.getByLabel("상태 필터")).toContainText("완료");
  await expect(page.getByLabel("상태 필터")).not.toContainText("검토 중");
  await expect(page.getByRole("heading", { name: /5월 14일/ })).toBeVisible();
  await expect(page.getByRole("heading", { name: /5월 13일/ })).toBeVisible();
  await expect(page.getByText("2026년 5월 15일")).toBeHidden();
  await expect(page.getByText("릴리즈 노트 정리")).toBeVisible();
  await expect(page.getByRole("heading", { name: "QA 승인" })).toBeVisible();
  await expect(page.getByText("오후 배포")).toBeVisible();
  await expect(page.getByText("오전 점검")).toBeHidden();
  await page.getByRole("button", { name: /1건 더 보기/ }).click();
  await expect(page.getByText("오전 점검")).toBeVisible();
  await expect(page.getByText("전날 회의")).toBeHidden();
  await page.getByRole("button", { name: /2026년 5월 13일/ }).click();
  await expect(page.getByText("전날 회의")).toBeVisible();
  await expect(page.getByText("Redis 점검")).toBeHidden();
  await expect(page.getByText("History: Redis 점검")).toBeHidden();
  await expect(page.getByText(/오전 01:00|01:00/)).toBeVisible();
  await expect(page.locator("span", { hasText: "Slack" }).first()).toBeVisible();

  await page.getByRole("button", { name: /2026년 5월 14일/ }).click();
  await expect(page.getByText("오전 점검")).toBeHidden();
  await expect(page.getByText("오후 배포")).toBeHidden();
  await expect(page.getByRole("heading", { name: /5월 14일/ })).toBeVisible();
  await expect(page.getByRole("heading", { name: /5월 13일/ })).toBeVisible();

  await page.getByRole("button", { name: /2026년 5월 14일/ }).click();
  await expect(page.getByText("오전 점검")).toBeHidden();
  await expect(page.getByText("Redis 점검")).toBeHidden();
  await page.getByRole("button", { name: /1건 더 보기/ }).click();
  await expect(page.getByText("오전 점검")).toBeVisible();
  await expect(page.getByText(/오전 01:00|01:00/)).toBeVisible();

  await expect(page.getByText("전날 회의")).toBeVisible();
  await expect(page.getByText("회의 완료")).toBeHidden();

  await page.getByRole("button", { name: "Open 오전 점검" }).click();
  await expect(page.getByText("Redis 점검")).toBeVisible();
  const sourceLink = page.getByRole("link", { name: "Open source" });
  await expect(sourceLink).toHaveAttribute("href", "https://slack.example/1");
  await expect(sourceLink).toHaveAttribute("target", "_blank");
  await expect(sourceLink).toHaveAttribute("rel", /noopener/);

  await page.context().route("https://slack.example/1", async (route) => {
    await route.fulfill({ contentType: "text/html", body: "<title>Slack source</title>" });
  });
  const popupPromise = page.waitForEvent("popup");
  await sourceLink.click();
  const popup = await popupPromise;
  expect(popup.url()).toBe("https://slack.example/1");
  await popup.close();
});

test("Timeline filters hide reviewing and Source while past Calendar items show completed", async ({ page }) => {
  await page.route("**/api/v1/auth/me", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: {
        user: {
          id: "demo-admin",
          email: "admin@paraworks.com",
          role: "admin",
          permission_levels: ["public", "internal"],
          name: "Admin",
          title: "Admin",
          department: "Platform",
        },
      },
    });
  });
  await page.route("**/api/v1/notifications", async (route) => {
    await route.fulfill({ contentType: "application/json", json: { unread_count: 0, notifications: [] } });
  });
  await page.route("**/api/v1/projects", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: {
        project_count: 1,
        hidden_project_count: 0,
        hidden_evidence_count: 0,
        projects: [
          {
            project_key: "calendar-project",
            name: "Calendar Project",
            summary: "Calendar evidence",
            source_types: ["calendar"],
            evidence_count: 1,
            permission_level: "internal",
            latest_timestamp: "2026-05-15T08:00:00+09:00",
            pending_review_count: 0,
            evidence: [],
            activity_items: [],
            timeline_items: [
              {
                id: "timeline_event:calendar-past",
                item_type: "timeline_event",
                title: "Past calendar meeting",
                summary: "Meeting already happened.",
                source_links: ["https://calendar.google.com/event?eid=past"],
                source_snippets: ["Calendar evidence"],
                confidence_score: 0.9,
                permission_level: "internal",
                review_status: "approved",
                created_at: "2026-05-15T08:00:00+09:00",
                occurred_at: "2026-05-15T08:00:00+09:00",
                evidence_reason: "Calendar event",
                project_key: "calendar-project",
              },
            ],
          },
        ],
      },
    });
  });
  await page.addInitScript(() => window.localStorage.setItem("paraworks-demo-user", "demo-admin"));

  await page.goto("/timeline");

  await expect(page.getByRole("heading", { name: "Past calendar meeting" })).toBeVisible();
  await expect(page.locator("#timeline-status-filter option")).toHaveText(["상태 전체", "승인됨", "완료"]);
  await expect(page.locator("#timeline-source-filter option")).toHaveText(["소스 전체", "Slack", "Gmail", "Drive", "Calendar"]);
  await expect(page.locator("span", { hasText: /^완료$/ })).toBeVisible();
  await page.locator("#timeline-status-filter").selectOption("완료");
  await expect(page.getByRole("heading", { name: "Past calendar meeting" })).toBeVisible();
});

test("Timeline opens on the first project that has approved timeline items", async ({ page }) => {
  await page.route("**/api/v1/auth/me", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: {
        user: {
          id: "demo-admin",
          email: "admin@paraworks.com",
          role: "admin",
          permission_levels: ["public", "internal"],
          name: "Admin",
          title: "Admin",
          department: "Platform",
        },
      },
    });
  });
  await page.route("**/api/v1/notifications", async (route) => {
    await route.fulfill({ contentType: "application/json", json: { unread_count: 0, notifications: [] } });
  });
  await page.route("**/api/v1/projects", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: {
        project_count: 2,
        hidden_project_count: 0,
        hidden_evidence_count: 0,
        projects: [
          {
            project_key: "project-empty",
            name: "빈 프로젝트",
            summary: "아직 승인 타임라인이 없습니다.",
            source_types: [],
            evidence_count: 0,
            permission_level: "internal",
            latest_timestamp: "",
            pending_review_count: 0,
            evidence: [],
            activity_items: [],
            timeline_items: [],
          },
          {
            project_key: "project-alpha",
            name: "Project Alpha",
            summary: "승인 타임라인이 있는 프로젝트입니다.",
            source_types: ["slack"],
            evidence_count: 0,
            permission_level: "internal",
            latest_timestamp: "2026-05-15T09:00:00Z",
            pending_review_count: 0,
            evidence: [],
            activity_items: [],
            timeline_items: [
              {
                id: "timeline_event:1",
                item_type: "timeline_event",
                title: "오전 점검",
                summary: "Redis 점검",
                source_links: ["https://slack.example/1"],
                source_snippets: ["점검"],
                confidence_score: 0.9,
                permission_level: "internal",
                review_status: "approved",
                created_at: "2026-05-15T09:00:00Z",
                occurred_at: "2026-05-14T01:00:00+09:00",
                evidence_reason: "승인된 항목",
                project_key: "project-alpha",
              },
            ],
          },
        ],
      },
    });
  });
  await page.addInitScript(() => window.localStorage.setItem("paraworks-demo-user", "demo-admin"));

  await page.goto("/timeline");

  await expect(page.getByRole("button", { name: "Project Alpha" })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByText("오전 점검")).toBeVisible();
  await expect(page.getByText("승인된 프로젝트 타임라인 항목이 아직 없습니다.")).toBeHidden();
});

test("Timeline keeps dense dates readable with recent expansion, month headers, and date index", async ({ page }) => {
  await page.route("**/api/v1/auth/me", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: {
        user: {
          id: "demo-admin",
          email: "admin@paraworks.com",
          role: "admin",
          permission_levels: ["public", "internal"],
          name: "Admin",
          title: "Admin",
          department: "Platform",
        },
      },
    });
  });
  await page.route("**/api/v1/notifications", async (route) => {
    await route.fulfill({ contentType: "application/json", json: { unread_count: 0, notifications: [] } });
  });
  await page.route("**/api/v1/projects", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: {
        project_count: 1,
        hidden_project_count: 0,
        hidden_evidence_count: 0,
        projects: [
          {
            project_key: "project-dense",
            name: "Dense Project",
            summary: "Many dates",
            source_types: ["slack"],
            evidence_count: 0,
            permission_level: "internal",
            latest_timestamp: "2026-05-15T09:00:00+09:00",
            pending_review_count: 0,
            evidence: [],
            activity_items: [],
            timeline_items: [
              {
                id: "timeline_event:recent",
                item_type: "timeline_event",
                title: "Recent sprint decision",
                summary: "Recent summary",
                source_links: ["https://slack.example/recent"],
                source_snippets: ["recent"],
                confidence_score: 0.9,
                permission_level: "internal",
                review_status: "approved",
                created_at: "2026-05-15T09:00:00+09:00",
                occurred_at: "2026-05-15T09:00:00+09:00",
                evidence_reason: "test",
                project_key: "project-dense",
              },
              {
                id: "timeline_event:old-may",
                item_type: "timeline_event",
                title: "Old May archive",
                summary: "Old May summary",
                source_links: ["https://slack.example/old-may"],
                source_snippets: ["old may"],
                confidence_score: 0.9,
                permission_level: "internal",
                review_status: "approved",
                created_at: "2026-05-01T09:00:00+09:00",
                occurred_at: "2026-05-01T09:00:00+09:00",
                evidence_reason: "test",
                project_key: "project-dense",
              },
              {
                id: "timeline_event:april",
                item_type: "timeline_event",
                title: "April archive",
                summary: "April summary",
                source_links: ["https://slack.example/april"],
                source_snippets: ["april"],
                confidence_score: 0.9,
                permission_level: "internal",
                review_status: "approved",
                created_at: "2026-04-20T09:00:00+09:00",
                occurred_at: "2026-04-20T09:00:00+09:00",
                evidence_reason: "test",
                project_key: "project-dense",
              },
            ],
          },
        ],
      },
    });
  });
  await page.addInitScript(() => window.localStorage.setItem("paraworks-demo-user", "demo-admin"));

  await page.goto("/timeline");

  await expect(page.locator("#timeline-status-filter option")).toHaveText(["상태 전체", "승인됨", "완료"]);
  await expect(page.getByTestId("timeline-date-index")).toBeVisible();
  const initialDateIndexBox = await page.getByTestId("timeline-date-index").boundingBox();
  const initialDateIndexStyle = await page.getByTestId("timeline-date-index").evaluate((element) => {
    const style = window.getComputedStyle(element);
    return { position: style.position, top: style.top };
  });
  expect(initialDateIndexStyle.position).toBe("sticky");
  expect(initialDateIndexStyle.top).not.toBe("auto");
  await expect(page.getByTestId("timeline-summary-strip")).toBeVisible();
  await expect(page.getByTestId("timeline-month-header-2026-05")).toBeVisible();
  await expect(page.getByTestId("timeline-month-header-2026-04")).toBeVisible();
  await expect(page.getByText("Recent sprint decision")).toBeVisible();
  await expect(page.getByText("Old May archive")).toBeHidden();
  await expect(page.getByText("April archive")).toBeHidden();
  await expect(page.getByTestId("timeline-date-index-2026-04-20")).toBeHidden();

  await page.getByTestId("timeline-date-index-2026-05-01").click();
  await expect(page.getByText("Old May archive")).toBeVisible();

  await page.getByTestId("timeline-month-nav-2026-04").click();
  await expect(page.getByTestId("timeline-date-index-2026-04-20")).toBeVisible();
  await page.getByTestId("timeline-date-index-2026-04-20").click();
  await expect(page.getByText("April archive")).toBeVisible();

  await page.getByTestId("timeline-date-density-toggle").click();
  await expect(page.getByTestId("timeline-date-index-2026-05-14")).toBeVisible();
  await page.evaluate(() => window.scrollTo(0, 700));
  const scrolledDateIndexBox = await page.getByTestId("timeline-date-index").boundingBox();
  expect(scrolledDateIndexBox?.y).toBeGreaterThanOrEqual(90);
  expect(scrolledDateIndexBox?.y).toBeLessThan(initialDateIndexBox?.y ?? 9999);
  expect(scrolledDateIndexBox?.y).toBeLessThanOrEqual(130);
});
