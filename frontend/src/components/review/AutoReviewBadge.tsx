import { Bot, UserRoundCheck } from "lucide-react";

export function AutoReviewBadge({ resolutionSource }: { resolutionSource?: "human" | "auto_policy" | null }) {
  if (resolutionSource === "auto_policy") {
    return <span className="badge blue" data-testid="trust-source-auto"><Bot className="mr-1 h-3.5 w-3.5" aria-hidden="true" />자동 검증</span>;
  }
  return <span className="badge green" data-testid="trust-source-human"><UserRoundCheck className="mr-1 h-3.5 w-3.5" aria-hidden="true" />사람 승인</span>;
}
