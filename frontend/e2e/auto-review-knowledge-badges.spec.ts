import { expect, test } from "@playwright/test";
import { AutoReviewBadge } from "../src/components/review/AutoReviewBadge";

test("shared trust badge maps only human and automatic public labels", () => {
  const human = AutoReviewBadge({ resolutionSource: "human" });
  const automatic = AutoReviewBadge({ resolutionSource: "auto_policy" });
  expect(human.props["data-testid"]).toBe("trust-source-human");
  expect(human.props.children.at(-1)).toBe("사람 승인");
  expect(automatic.props["data-testid"]).toBe("trust-source-auto");
  expect(automatic.props.children.at(-1)).toBe("자동 검증");
});
