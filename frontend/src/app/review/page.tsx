"use client";

import {
  Bot,
  CheckCircle2,
  CheckSquare,
  Coins,
  FileSearch,
  Pencil,
  RefreshCw,
  Sparkles,
  Square,
  XCircle,
  Layers,
} from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, type MouseEvent, useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ReviewWorkflowContextPanel } from "@/app/review/ReviewWorkflowContextPanel";
import { AutoReviewTrustPanel } from "@/app/review/AutoReviewTrustPanel";
import { deriveReviewBulkProjectState, parseReviewWorkflowQuery } from "@/app/review/reviewWorkflowContext";
import { SourceEvidenceDrawer } from "@/components/shared/SourceEvidenceDrawer";
import { apiGet, apiPatch, apiPost } from "@/lib/api/client";
import { getReviewWorkflowStatus, resumeReviewWorkflow } from "@/lib/api/reviewWorkflow";
import { notifyReviewQueueUpdated } from "@/lib/reviewQueueEvents";
import { sourceFamilyLabel, sourceTypeFromUrl } from "@/lib/sourceLabels";
import type {
  ReviewItem,
  ReviewGroup,
  ReviewEvidenceRequest,
  ReviewApprovalResponse,
  ReviewBulkActionResponse,
  ReviewItemUpdate,
  ReviewPromotionPreview,
  ReviewResponse,
  ReviewPromotionResult,
  ReviewWorkflowStatus,
} from "@/lib/api/types";

const REVIEW_PAGE_SIZE = 50;

const LOW_SIGNAL_REVIEW_TITLES = new Set(["paraworks source 연결", "source 연결", "untitled", "unknown"]);
const DISPLAY_TITLE_KEYS = [
  "title",
  "summary",
  "decision_summary",
  "reason",
  "priority_reason",
  "task_summary",
  "source_title",
  "project_assignment_summary",
  "evidence_reason",
  "recommended_next_step",
] as const;

function stringField(value: unknown) {
  return typeof value === "string" ? value : "";
}

function cleanReviewDisplayText(value: string) {
  const withoutTags = value.replace(/\\n/g, " ").replace(/<[^>]+>/g, " ");
  const cleaned = withoutTags.split(/\s+/).join(" ").trim();
  const withoutSourcePrefix = cleaned.replace(/^(?:Google Drive file changed|Gmail attachment):\s*/i, "").trim();
  const metadataMatch = withoutSourcePrefix.match(/(?:^|\s)(?:Description|Location|Start|End|Marker|From|Date|Mime type|Owner|Last modifier|Modified|Parent subject|Attachment size):\s/i);
  if (metadataMatch?.index !== undefined) {
    return withoutSourcePrefix.slice(0, metadataMatch.index).trim();
  }
  return withoutSourcePrefix;
}

function knownStringField(value: unknown) {
  const text = cleanReviewDisplayText(stringField(value));
  if (!text || text.toLowerCase() === "unknown") return "";
  return text;
}

function numberField(value: unknown) {
  return typeof value === "number" ? value : undefined;
}

function needsProjectSelection(item: ReviewItem, preview?: ReviewPromotionPreview) {
  return (
    item.payload.project_assignment_method === "llm_tool" &&
    (!stringField(item.payload.project_key) ||
      item.payload.project_needs_user_selection === true ||
      preview?.missing_required_fields?.includes("project_key"))
  );
}


function itemTitle(item: ReviewItem | ReviewGroup) {
  if ('payload' in item) {
    return displayTitleFromPayload(item.payload, item.id);
  }
  return item.title;
}

function displayTitleFromPayload(payload: ReviewItem["payload"], itemId: number) {
  const title = cleanReviewDisplayText(stringField(payload.title));
  if (title && !isLowSignalReviewTitle(title)) return title;
  for (const key of DISPLAY_TITLE_KEYS.slice(1)) {
    const value = cleanReviewDisplayText(stringField(payload[key]));
    if (value && !isLowSignalReviewTitle(value)) return value.split(/\s+/).join(" ");
  }
  return `Review item ${itemId}`;
}

function isLowSignalReviewTitle(value: string) {
  const normalized = value.split(/\s+/).join(" ").toLowerCase();
  return LOW_SIGNAL_REVIEW_TITLES.has(normalized) || normalized.endsWith(" source 연결");
}

function summaryKey(item: ReviewItem) {
  if (typeof item.payload.summary === "string") return "summary";
  if (typeof item.payload.decision_summary === "string") return "decision_summary";
  if (typeof item.payload.reason === "string") return "reason";
  if (typeof item.payload.priority_reason === "string") return "priority_reason";
  return "summary";
}

function itemSummary(item: ReviewItem) {
  const summary = cleanReviewDisplayText(stringField(item.payload[summaryKey(item)]));
  return summary || "요약을 생성하지 못했습니다. 근거를 확인한 뒤 수정하거나 추가 근거를 요청하세요.";
}

function mailDocsWorkFields(item: ReviewItem) {
  const agentName = stringField(item.payload.agent_name);
  if (agentName !== "mail_document_agent") return undefined;
  const nextStep = stringField(item.payload.recommended_next_step);
  const taskSummary = stringField(item.payload.task_summary);
  const businessContext = stringField(item.payload.business_context);
  if (!nextStep && !taskSummary && !businessContext) return undefined;
  return {
    nextStep,
    taskSummary,
    businessContext,
    assignee: stringField(item.payload.assignee),
    dueDate: stringField(item.payload.due_date),
    counterparty: stringField(item.payload.counterparty),
  };
}

function projectRoutingLabel(item: ReviewItem) {
  return item.payload.project_assignment_method === "llm_tool"
    ? "LLM 프로젝트 분류"
    : undefined;
}

function projectRoutingSummary(item: ReviewItem) {
  const summary = stringField(item.payload.project_assignment_summary).trim();
  return summary || undefined;
}

function projectRoutingReason(item: ReviewItem) {
  const reason = stringField(item.payload.project_assignment_reason).trim();
  return reason || undefined;
}

function projectAssignmentFields(item: ReviewItem) {
  if (item.item_type !== "project_assignment") return undefined;
  const projectName = cleanReviewDisplayText(stringField(item.payload.project_name));
  const taskSummary = cleanReviewDisplayText(stringField(item.payload.task_summary)) || itemSummary(item);
  const evidenceReason = cleanReviewDisplayText(stringField(item.payload.evidence_reason));
  const sourceTitle = cleanReviewDisplayText(stringField(item.payload.source_title));
  const sourceType = cleanReviewDisplayText(stringField(item.payload.source_type));
  if (!projectName && !taskSummary && !evidenceReason && !sourceTitle && !sourceType) return undefined;
  return {
    projectName,
    taskSummary,
    evidenceReason,
    sourceTitle,
    sourceType,
  };
}

function agentDisplayName(agentName: string, item?: ReviewItem) {
  if (agentName === "mail_document_agent" && item) {
    return mailDocumentSourceLabel(item) || "Mail/Docs Agent";
  }
  const labels: Record<string, string> = {
    project_classifier: "프로젝트 분류기",
    slack_agent: "Slack Agent",
    mail_document_agent: "Mail/Docs Agent",
    memory_extraction_agent: "Memory Agent",
  };
  return labels[agentName] ?? agentName;
}

function mailDocumentSourceLabel(item: ReviewItem) {
  return sourceFamilyLabel(sourceTypesForItem(item));
}

function sourceTypesForItem(item: ReviewItem) {
  const evidenceTypes = (item.source_evidence ?? []).map((row) => row.source_type);
  const payloadTypes = Array.isArray(item.payload.source_types) ? item.payload.source_types : [];
  const payloadType = stringField(item.payload.source_type);
  const urlTypes = (item.source_links ?? []).map((link) => sourceTypeFromUrl(link));
  return [...evidenceTypes, ...payloadTypes, payloadType, ...urlTypes];
}

type ReviewSourceType = "slack" | "calendar" | "drive" | "gmail" | "gmail_attachment" | "";

function normalizeReviewSourceType(value: unknown): ReviewSourceType {
  if (typeof value !== "string") return "";
  const normalized = value.trim().toLowerCase();
  if (
    normalized === "slack" ||
    normalized === "calendar" ||
    normalized === "drive" ||
    normalized === "gmail" ||
    normalized === "gmail_attachment"
  ) {
    return normalized;
  }
  return "";
}

function firstReviewSourceType(values: unknown[]): ReviewSourceType {
  return values.map(normalizeReviewSourceType).find(Boolean) || "";
}

function primarySourceType(item: ReviewItem) {
  const payloadType = firstReviewSourceType([item.payload.source_type]);
  if (payloadType) return payloadType;
  const payloadTypes = Array.isArray(item.payload.source_types) ? firstReviewSourceType(item.payload.source_types) : "";
  if (payloadTypes) return payloadTypes;
  const evidenceType = firstReviewSourceType((item.source_evidence ?? []).map((row) => row.source_type));
  if (evidenceType) return evidenceType;
  return firstReviewSourceType((item.source_links ?? []).map((link) => sourceTypeFromUrl(link)));
}

function agentBadgeLabel(item: ReviewItem, agentName: string) {
  if (agentName === "mail_document_agent" && Array.isArray(item.payload.source_types)) {
    return mailDocumentSourceLabel(item) || "Mail/Docs Agent";
  }
  const sourceType = primarySourceType(item);
  if (sourceType === "slack" || agentName === "slack_agent") return "Slack Agent";
  if (sourceType === "calendar") return "Calendar Agent";
  if (sourceType === "drive") return "Google Drive Agent";
  if (sourceType === "gmail" || sourceType === "gmail_attachment") return "Mail Agent";
  return agentDisplayName(agentName, item);
}

function agentBadgeClass(item: ReviewItem, agentName: string) {
  const sourceType = primarySourceType(item);
  if (sourceType === "slack" || agentName === "slack_agent") return "border border-violet-200 bg-violet-100/80 text-violet-700";
  if (sourceType === "calendar") return "border border-emerald-200 bg-emerald-100/80 text-emerald-700";
  if (sourceType === "drive") return "border border-blue-200 bg-blue-100/80 text-blue-700";
  if (sourceType === "gmail" || sourceType === "gmail_attachment") return "border border-rose-200 bg-rose-100/80 text-rose-700";
  return "border border-slate-200 bg-slate-100 text-slate-700";
}

function routeLabel(route: string) {
  if (route === "/projects") return "프로젝트에서 보기";
  if (route === "/timeline") return "타임라인에서 보기";
  if (route === "/knowledge") return "지식 보관함에서 보기";
  return `${route} 열기`;
}

function itemTypeLabel(itemType: string) {
  const labels: Record<string, string> = {
    decision_record: "결정 기록",
    history_event: "히스토리",
    timeline_event: "타임라인",
    todo: "할 일",
    message_review: "메시지 검토",
    project_assignment: "프로젝트 연결",
  };
  return labels[itemType] ?? itemType.replaceAll("_", " ");
}

function formatCost(value: unknown) {
  const cost = numberField(value);
  if (cost === undefined) return undefined;
  return `$${cost.toFixed(6)}`;
}

function safeMutationError(error: unknown, fallback: string) {
  if (error instanceof Error && (error.message.includes("Authentication required") || error.message.includes("401"))) {
    return error.message;
  }
  return fallback;
}

function reviewPromptLabel(item: ReviewItem, agentName: string) {
  const payloadPrompt = knownStringField(item.payload.prompt_version);
  if (payloadPrompt) return payloadPrompt;

  const agentRunPrompt = knownStringField(item.agent_run_details?.prompt_version);
  if (agentRunPrompt) return agentRunPrompt;

  if (agentName === "project_classifier") return "규칙 기반 프로젝트 연결";
  if (!item.agent_run_id) return "규칙 기반 처리";
  return "프롬프트 정보 없음";
}

function reviewCostLabel(item: ReviewItem, agentName: string) {
  const payloadCost = formatCost(item.payload.estimated_cost_usd);
  if (payloadCost) return payloadCost;

  const agentRunCost = numberField(item.agent_run_details?.estimated_cost_usd);
  if (agentRunCost !== undefined && item.agent_run_id) return formatCost(agentRunCost) ?? "$0.000000";

  if (agentName === "project_classifier") return "추가 LLM 비용 없음";
  if (!item.agent_run_id) return "LLM 미사용";
  return "비용 정보 없음";
}

type PromotionNotice = {
  itemTitle: string;
  result: ReviewPromotionResult;
};

type BulkConfirmState = {
  action: "approve" | "reject";
  itemIds: number[];
  scope: "selected" | "loaded" | "similar";
};

type ReviewContextMenu = {
  x: number;
  y: number;
  item: ReviewItem;
};

type ReviewQueryContext = {
  generation: number;
  key: string;
};

export default function ReviewPage() {
  return (
    <Suspense fallback={<ReviewPageFallback />}>
      <ReviewPageContent />
    </Suspense>
  );
}

function ReviewPageFallback() {
  return (
    <div className="reference-dashboard space-y-5">
      <div className="rounded-lg border border-[var(--line-soft)] bg-[var(--glass-elevated)] p-8 text-center text-sm text-[var(--ink-muted)]">
        검토 항목을 불러오는 중입니다...
      </div>
    </div>
  );
}

function ReviewPageContent() {
  const [groups, setGroups] = useState<ReviewGroup[]>([]);
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({});
  const [editingId, setEditingId] = useState<number>();
  const [editTitle, setEditTitle] = useState("");
  const [editSummary, setEditSummary] = useState("");
  const [editProjectKey, setEditProjectKey] = useState("");
  const [definedProjects, setDefinedProjects] = useState<Array<{project_key: string, name: string}>>([]);
  const [editNextStep, setEditNextStep] = useState("");
  const [evidenceRequestId, setEvidenceRequestId] = useState<number>();
  const [evidenceRequestNote, setEvidenceRequestNote] = useState("");
  const [previews, setPreviews] = useState<Record<number, ReviewPromotionPreview>>({});
  const [pendingAction, setPendingAction] = useState<string>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();
  const [promotionNotice, setPromotionNotice] = useState<PromotionNotice>();
  const [totalCount, setTotalCount] = useState(0);
  const [loadedOffset, setLoadedOffset] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [selectedItemIds, setSelectedItemIds] = useState<Set<number>>(() => new Set());
  const [bulkProjectKey, setBulkProjectKey] = useState("");
  const [bulkConfirm, setBulkConfirm] = useState<BulkConfirmState>();
  const [contextMenu, setContextMenu] = useState<ReviewContextMenu>();
  const [deepLinkedItemId, setDeepLinkedItemId] = useState<number>();
  const [workflowStatus, setWorkflowStatus] = useState<ReviewWorkflowStatus>();
  const [workflowLoading, setWorkflowLoading] = useState(false);
  const [workflowUnavailable, setWorkflowUnavailable] = useState(false);
  const [workflowActionPending, setWorkflowActionPending] = useState(false);
  const [reviewView, setReviewView] = useState<"pending" | "auto">("pending");
  const searchParams = useSearchParams();
  const workflowQuery = parseReviewWorkflowQuery(searchParams.get("workflow_thread_id"));
  const workflowThreadId = workflowQuery.kind === "workflow" ? workflowQuery.workflowThreadId : undefined;
  const invalidWorkflowContext = workflowQuery.kind === "invalid";
  const itemId = Number(searchParams.get("itemId") ?? searchParams.get("item_id"));
  const deepLinkItemId = Number.isInteger(itemId) && itemId > 0 ? itemId : undefined;
  const workflowQueryKey = workflowQuery.kind === "workflow" ? `workflow:${workflowThreadId}` : workflowQuery.kind;
  const queryKey = `${workflowQueryKey}:${deepLinkItemId ?? ""}:${reviewView}`;
  const [renderedContextKey, setRenderedContextKey] = useState(queryKey);
  const contextGeneration = useRef(0);
  const listRequestGeneration = useRef(0);
  const statusRequestGeneration = useRef(0);
  const currentQueryKey = useRef(queryKey);

  // A layout effect updates the async-action guard before the browser can paint
  // the next query context, without mutating a ref while React is rendering.
  useLayoutEffect(() => {
    currentQueryKey.current = queryKey;
  }, [queryKey]);

  function captureQueryContext(): ReviewQueryContext {
    return { generation: contextGeneration.current, key: queryKey };
  }

  function isCurrentQueryContext(context: ReviewQueryContext) {
    return context.generation === contextGeneration.current && context.key === currentQueryKey.current;
  }

  const loadItems = useCallback(async (
    nextOffset = 0,
    append = false,
    requestContext = contextGeneration.current,
    requestKey = queryKey,
  ) => {
    if (invalidWorkflowContext) return;
    if (requestContext !== contextGeneration.current || requestKey !== currentQueryKey.current) return;
    const request = ++listRequestGeneration.current;
    if (requestContext === contextGeneration.current) {
      setLoading(true);
    }

    try {
      const params = new URLSearchParams({
        status: reviewView === "auto" ? "approved" : "pending_review",
        limit: String(REVIEW_PAGE_SIZE),
        offset: String(nextOffset),
        include_previews: "false",
      });
      if (reviewView === "auto") params.set("resolution_source", "auto_policy");
      if (workflowThreadId) params.set("workflow_thread_id", workflowThreadId);
      if (reviewView === "pending") void apiGet<{projects: Array<{project_key: string, name: string}>}>("/api/v1/projects/defined")
        .then((projectsRes) => {
          if (
            requestContext === contextGeneration.current
            && requestKey === currentQueryKey.current
            && request === listRequestGeneration.current
          ) {
            setDefinedProjects(projectsRes.projects || []);
          }
        })
        .catch(() => {
          if (
            requestContext === contextGeneration.current
            && requestKey === currentQueryKey.current
            && request === listRequestGeneration.current
          ) {
            setDefinedProjects([]);
          }
        });
      const review = await apiGet<ReviewResponse>(`/api/v1/review?${params.toString()}`);
      if (
        requestContext !== contextGeneration.current
        || requestKey !== currentQueryKey.current
        || request !== listRequestGeneration.current
      ) return;
      setGroups((current) => (append ? mergeReviewGroups(current, review.groups || []) : review.groups || []));
      setTotalCount(review.total_count ?? review.items.length);
      setLoadedOffset(nextOffset);
      setHasMore(Boolean(review.has_more));
    } catch (caught) {
      if (
        requestContext !== contextGeneration.current
        || requestKey !== currentQueryKey.current
        || request !== listRequestGeneration.current
      ) return;
      setError(caught instanceof Error ? caught.message : "검토 항목을 불러오지 못했습니다.");
    } finally {
      if (
        requestContext === contextGeneration.current
        && requestKey === currentQueryKey.current
        && request === listRequestGeneration.current
      ) {
        setLoading(false);
      }
    }
  }, [invalidWorkflowContext, queryKey, reviewView, workflowThreadId]);

  const loadWorkflowStatus = useCallback(async (
    requestContext = contextGeneration.current,
    requestKey = queryKey,
  ) => {
    if (invalidWorkflowContext) return;
    if (!workflowThreadId) return;
    if (requestContext !== contextGeneration.current || requestKey !== currentQueryKey.current) return;
    const request = ++statusRequestGeneration.current;
    if (requestContext === contextGeneration.current) {
      setWorkflowLoading(true);
      setWorkflowUnavailable(false);
    }
    try {
      const status = await getReviewWorkflowStatus(workflowThreadId);
      if (
        requestContext !== contextGeneration.current
        || requestKey !== currentQueryKey.current
        || request !== statusRequestGeneration.current
      ) return;
      setWorkflowStatus(status);
    } catch {
      if (
        requestContext !== contextGeneration.current
        || requestKey !== currentQueryKey.current
        || request !== statusRequestGeneration.current
      ) return;
      setWorkflowStatus(undefined);
      setWorkflowUnavailable(true);
    } finally {
      if (
        requestContext === contextGeneration.current
        && requestKey === currentQueryKey.current
        && request === statusRequestGeneration.current
      ) {
        setWorkflowLoading(false);
      }
    }
  }, [invalidWorkflowContext, queryKey, workflowThreadId]);

  useEffect(() => {
    const requestContext = ++contextGeneration.current;
    listRequestGeneration.current += 1;
    statusRequestGeneration.current += 1;
    setGroups([]);
    setExpandedGroups({});
    setDefinedProjects([]);
    setPreviews({});
    setSelectedItemIds(new Set());
    setBulkProjectKey("");
    setBulkConfirm(undefined);
    setContextMenu(undefined);
    setPendingAction(undefined);
    setEditingId(undefined);
    setEditTitle("");
    setEditSummary("");
    setEditProjectKey("");
    setEditNextStep("");
    setEvidenceRequestId(undefined);
    setEvidenceRequestNote("");
    setPromotionNotice(undefined);
    setError(undefined);
    setTotalCount(0);
    setLoadedOffset(0);
    setHasMore(false);
    setDeepLinkedItemId(deepLinkItemId);
    setWorkflowStatus(undefined);
    setWorkflowUnavailable(false);
    setWorkflowLoading(Boolean(workflowThreadId));
    setWorkflowActionPending(false);
    setRenderedContextKey(queryKey);
    if (invalidWorkflowContext) return;
    void loadItems(0, false, requestContext, queryKey);
    if (workflowThreadId) void loadWorkflowStatus(requestContext, queryKey);
  }, [deepLinkItemId, invalidWorkflowContext, loadItems, loadWorkflowStatus, queryKey, workflowThreadId]);

  const refreshReviewData = useCallback(async (context: ReviewQueryContext) => {
    if (!isCurrentQueryContext(context)) return;
    await Promise.all([
      loadItems(0, false, context.generation, context.key),
      loadWorkflowStatus(context.generation, context.key),
    ]);
  }, [loadItems, loadWorkflowStatus]);

  useEffect(() => {
    if (renderedContextKey !== queryKey || !deepLinkedItemId || groups.length === 0) return;
    const targetGroup = groups.find((group) => group.items.some((item) => item.id === deepLinkedItemId));
    if (!targetGroup) return;
    setExpandedGroups((current) => ({ ...current, [targetGroup.group_id]: true }));
    const timer = window.setTimeout(() => {
      document.getElementById(`review-item-${deepLinkedItemId}`)?.scrollIntoView({ block: "center", behavior: "smooth" });
    }, 50);
    return () => window.clearTimeout(timer);
  }, [deepLinkItemId, deepLinkedItemId, groups, queryKey, renderedContextKey]);

  useEffect(() => {
    if (!contextMenu) return;
    const closeMenu = () => setContextMenu(undefined);
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") closeMenu();
    };
    window.addEventListener("click", closeMenu);
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      window.removeEventListener("click", closeMenu);
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [contextMenu]);

  async function loadPreviewsForItems(items: ReviewItem[]) {
    const missingItems = items.filter((item) => !previews[item.id]);
    if (missingItems.length === 0) return;
    const requestContext = contextGeneration.current;
    const previewPairs = await Promise.all(
      missingItems.map(async (item) => [
        item.id,
        await apiGet<ReviewPromotionPreview>(`/api/v1/review/${item.id}/promotion-preview`),
      ] as const),
    );
    if (requestContext !== contextGeneration.current) return;
    setPreviews((current) => ({ ...current, ...Object.fromEntries(previewPairs) }));
  }

  const toggleGroup = (group: ReviewGroup) => {
    const willOpen = !expandedGroups[group.group_id];
    setExpandedGroups(prev => ({ ...prev, [group.group_id]: willOpen }));
    if (willOpen && reviewView === "pending") void loadPreviewsForItems(group.items || []);
  };

  function startEdit(item: ReviewItem) {
    setEditingId(item.id);
    setEditTitle(itemTitle(item));
    setEditSummary(itemSummary(item));
    setEditProjectKey(stringField(item.payload.project_key));
    setEditNextStep(stringField(item.payload.recommended_next_step));
    setError(undefined);
  }

  async function runStatusAction(
    item: ReviewItem,
    action: "approve" | "reject" | "request-more-evidence",
    body?: ReviewEvidenceRequest,
  ) {
    const actionKey = `${item.id}:${action}`;
    const actionContext = captureQueryContext();
    setPendingAction(actionKey);
    setError(undefined);

    try {
      const result = await apiPost<ReviewApprovalResponse | ReviewItem>(`/api/v1/review/${item.id}/${action}`, body);
      if (!isCurrentQueryContext(actionContext)) return;
      if (action === "approve" && "promotion_result" in result && result.promotion_result) {
        setPromotionNotice({
          itemTitle: itemTitle(item),
          result: result.promotion_result,
        });
      }
      notifyReviewQueueUpdated();

      if (action === "request-more-evidence") {
        setEvidenceRequestId(undefined);
        setEvidenceRequestNote("");
      }
    } catch (caught) {
      if (isCurrentQueryContext(actionContext)) {
        setError(safeMutationError(caught, "검토 상태를 변경하지 못했습니다. 현재 상태를 다시 확인해 주세요."));
      }
    } finally {
      if (isCurrentQueryContext(actionContext)) {
        await refreshReviewData(actionContext);
        if (isCurrentQueryContext(actionContext)) setPendingAction(undefined);
      }
    }
  }

  async function updateItemProject(item: ReviewItem, projectKey: string) {
    const update: ReviewItemUpdate = {
      payload: {
        project_key: projectKey,
        project_needs_user_selection: false,
      },
    };
    const actionContext = captureQueryContext();
    setPendingAction(`${item.id}:project`);
    setError(undefined);
    try {
      await apiPatch<ReviewItem>(`/api/v1/review/${item.id}`, update);
      if (!isCurrentQueryContext(actionContext)) return;
      setPreviews((current) => {
        const next = { ...current };
        delete next[item.id];
        return next;
      });
      notifyReviewQueueUpdated();
    } catch (caught) {
      if (isCurrentQueryContext(actionContext)) {
        setError(safeMutationError(caught, "프로젝트를 저장하지 못했습니다. 현재 상태를 다시 확인해 주세요."));
      }
    } finally {
      if (isCurrentQueryContext(actionContext)) {
        await refreshReviewData(actionContext);
        if (isCurrentQueryContext(actionContext)) setPendingAction(undefined);
      }
    }
  }

  function openBulkConfirm(action: "approve" | "reject", itemIds: number[], scope: BulkConfirmState["scope"]) {
    if (itemIds.length === 0) return;
    setBulkConfirm({ action, itemIds, scope });
  }

  async function executeBulkAction(confirmState: BulkConfirmState) {
    const { action, itemIds } = confirmState;
    const label = action === "approve" ? "승인" : "반려";
    const actionContext = captureQueryContext();
    setPendingAction(`bulk:${action}`);
    setError(undefined);
    try {
      if (bulkProjectKey) {
        await Promise.all(
          itemIds.map((itemId) =>
            apiPatch<ReviewItem>(`/api/v1/review/${itemId}`, {
              payload: {
                project_key: bulkProjectKey,
                project_needs_user_selection: false,
              },
            }),
          ),
        );
      }
      const result = await apiPost<ReviewBulkActionResponse>("/api/v1/review/bulk", {
        action,
        item_ids: itemIds,
      });
      if (!isCurrentQueryContext(actionContext)) return;
      notifyReviewQueueUpdated();
      setSelectedItemIds((current) => {
        const next = new Set(current);
        for (const itemId of itemIds) next.delete(itemId);
        return next;
      });
      setBulkConfirm(undefined);
      if (result.failed_items.length > 0) {
        setError(`${label} 처리 중 ${result.failed_items.length}개 항목은 건너뛰었습니다. 필수 정보와 근거를 확인해 주세요.`);
      }
    } catch (caught) {
      if (isCurrentQueryContext(actionContext)) {
        setError(safeMutationError(caught, `선택한 항목을 ${label} 처리하지 못했습니다. 현재 상태를 다시 확인해 주세요.`));
      }
    } finally {
      if (isCurrentQueryContext(actionContext)) {
        await refreshReviewData(actionContext);
        if (isCurrentQueryContext(actionContext)) setPendingAction(undefined);
      }
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async function runBulkAction(action: "approve" | "reject") {
    const itemIds = groups.flatMap((group) => group.items.map((item) => item.id));
    if (itemIds.length === 0) return;
    const label = action === "approve" ? "승인" : "반려";
    if (!window.confirm(`현재 로드된 검토 항목 ${itemIds.length}개를 모두 ${label}할까요?`)) return;

    const actionContext = captureQueryContext();
    setPendingAction(`bulk:${action}`);
    setError(undefined);
    try {
      const result = await apiPost<ReviewBulkActionResponse>("/api/v1/review/bulk", {
        action,
        item_ids: itemIds,
      });
      if (!isCurrentQueryContext(actionContext)) return;
      notifyReviewQueueUpdated();
      if (result.failed_items.length > 0) {
        setError(`${label} 처리 중 ${result.failed_items.length}개 항목은 건너뛰었습니다. 필수 정보와 근거를 확인해 주세요.`);
      }
    } catch (caught) {
      if (isCurrentQueryContext(actionContext)) {
        setError(safeMutationError(caught, `모두 ${label} 처리하지 못했습니다. 현재 상태를 다시 확인해 주세요.`));
      }
    } finally {
      if (isCurrentQueryContext(actionContext)) {
        await refreshReviewData(actionContext);
        if (isCurrentQueryContext(actionContext)) setPendingAction(undefined);
      }
    }
  }
  async function saveEdit(item: ReviewItem) {
    const key = summaryKey(item);
    const update: ReviewItemUpdate = {
      payload: {
        ...item.payload,
        title: editTitle,
        [key]: editSummary,
        project_key: editProjectKey,
        ...(mailDocsWorkFields(item) ? { recommended_next_step: editNextStep } : {}),
      },
    };

    const actionContext = captureQueryContext();
    setPendingAction(`${item.id}:edit`);
    setError(undefined);

    try {
      await apiPatch<ReviewItem>(`/api/v1/review/${item.id}`, update);
      if (!isCurrentQueryContext(actionContext)) return;
      notifyReviewQueueUpdated();
      setEditingId(undefined);
    } catch (caught) {
      if (isCurrentQueryContext(actionContext)) {
        setError(safeMutationError(caught, "검토 항목을 수정하지 못했습니다. 현재 상태를 다시 확인해 주세요."));
      }
    } finally {
      if (isCurrentQueryContext(actionContext)) {
        await refreshReviewData(actionContext);
        if (isCurrentQueryContext(actionContext)) setPendingAction(undefined);
      }
    }
  }

  async function resumeWorkflow() {
    if (!workflowThreadId || !workflowStatus) return;
    const allowed = workflowStatus.status === "awaiting_human_review"
      ? workflowStatus.review_resolution_ready && workflowStatus.resume_allowed
      : workflowStatus.status === "checkpoint_failed" && workflowStatus.retry_allowed;
    if (!allowed) return;

    const actionContext = captureQueryContext();
    setWorkflowActionPending(true);
    try {
      await resumeReviewWorkflow(workflowThreadId);
    } catch {
      // A lost resume response can still have advanced the checkpoint; reload its authoritative status.
    } finally {
      if (isCurrentQueryContext(actionContext)) {
        await refreshReviewData(actionContext);
        if (isCurrentQueryContext(actionContext)) setWorkflowActionPending(false);
      }
    }
  }

  const isRenderedContextCurrent = renderedContextKey === queryKey;
  const visibleGroups = isRenderedContextCurrent ? groups : [];
  const visibleError = isRenderedContextCurrent ? error : undefined;
  const visibleWorkflowStatus = isRenderedContextCurrent ? workflowStatus : undefined;
  const visibleWorkflowUnavailable = isRenderedContextCurrent && workflowUnavailable;
  const visibleWorkflowLoading = isRenderedContextCurrent ? workflowLoading : Boolean(workflowThreadId);
  const visibleLoading = isRenderedContextCurrent ? loading : true;
  const visibleTotalCount = isRenderedContextCurrent ? totalCount : 0;
  const visibleHasMore = isRenderedContextCurrent && hasMore;
  const bulkProjectState = deriveReviewBulkProjectState({
    isRenderedContextCurrent,
    loading: visibleLoading,
    invalidWorkflowContext,
    bulkProjectKey,
    definedProjects,
  });
  const totalAgentItems = visibleGroups.reduce((acc, g) => acc + g.items.filter(i => Boolean(i.payload.agent_name)).length, 0);
  const loadedItemCount = visibleGroups.reduce((acc, group) => acc + group.items.length, 0);
  const loadedItems = visibleGroups.flatMap((group) => group.items);
  const loadedItemIds = loadedItems.map((item) => item.id);
  const selectedLoadedIds = loadedItemIds.filter((itemId) => selectedItemIds.has(itemId));
  const duplicateItemIds = visibleGroups
    .filter((group) => group.total_count > 1)
    .flatMap((group) => group.items.map((item) => item.id));
  const allLoadedSelected = loadedItemCount > 0 && selectedLoadedIds.length === loadedItemCount;
  const someLoadedSelected = selectedLoadedIds.length > 0 && !allLoadedSelected;
  const authRequired = visibleError ? visibleError.includes("Authentication required") || visibleError.includes("401") : false;

  function toggleAllLoadedSelection() {
    setSelectedItemIds((current) => {
      const next = new Set(current);
      if (allLoadedSelected) {
        for (const itemId of loadedItemIds) next.delete(itemId);
      } else {
        for (const itemId of loadedItemIds) next.add(itemId);
      }
      return next;
    });
  }

  function toggleItemSelection(itemId: number) {
    setSelectedItemIds((current) => {
      const next = new Set(current);
      if (next.has(itemId)) next.delete(itemId);
      else next.add(itemId);
      return next;
    });
  }

  function groupItemIds(group: ReviewGroup) {
    return group.items.map((item) => item.id);
  }

  function isGroupFullySelected(group: ReviewGroup) {
    const itemIds = groupItemIds(group);
    return itemIds.length > 0 && itemIds.every((itemId) => selectedItemIds.has(itemId));
  }

  function isGroupPartiallySelected(group: ReviewGroup) {
    const itemIds = groupItemIds(group);
    return itemIds.some((itemId) => selectedItemIds.has(itemId)) && !isGroupFullySelected(group);
  }

  function toggleGroupSelection(event: MouseEvent, group: ReviewGroup) {
    event.stopPropagation();
    const itemIds = groupItemIds(group);
    const shouldClear = isGroupFullySelected(group);
    setSelectedItemIds((current) => {
      const next = new Set(current);
      for (const itemId of itemIds) {
        if (shouldClear) next.delete(itemId);
        else next.add(itemId);
      }
      return next;
    });
  }

  function openContextMenu(event: MouseEvent, item: ReviewItem) {
    event.preventDefault();
    const menuWidth = 240;
    const menuHeight = 172;
    const viewportPadding = 12;
    const x = Math.min(event.clientX, window.innerWidth - menuWidth - viewportPadding);
    const y = Math.min(event.clientY, window.innerHeight - menuHeight - viewportPadding);
    setContextMenu({ x: Math.max(viewportPadding, x), y: Math.max(viewportPadding, y), item });
  }

  return (
    <div className="reference-dashboard space-y-5">
      <div className="space-y-3">
        <div className="page-heading reference-heading">
          <div>
            <p className="text-[13px] font-bold text-[var(--primary-dark)]">Review Items</p>
            <h1>검토사항</h1>
            <p>
              유사한 항목들은 그룹화되어 표시됩니다. 각 그룹을 펼쳐 상세 내용을 확인하세요.
            </p>
          </div>
        </div>
        <ReviewWorkflowContextPanel
          status={visibleWorkflowStatus}
          loading={visibleWorkflowLoading}
          unavailable={visibleWorkflowUnavailable}
          invalidContext={invalidWorkflowContext}
          actionPending={workflowActionPending}
          onResume={() => void resumeWorkflow()}
        />
        <div className="inline-flex rounded-lg border border-[var(--line-soft)] bg-white p-1" aria-label="검토 보기">
          <button type="button" aria-pressed={reviewView === "pending"} onClick={() => setReviewView("pending")} className={`h-8 rounded-md px-3 text-sm font-semibold ${reviewView === "pending" ? "bg-[#21132b] text-white" : "text-[var(--ink-muted)]"}`}>검토 대기</button>
          <button type="button" aria-pressed={reviewView === "auto"} onClick={() => setReviewView("auto")} className={`h-8 rounded-md px-3 text-sm font-semibold ${reviewView === "auto" ? "bg-[#21132b] text-white" : "text-[var(--ink-muted)]"}`}>자동 승인</button>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex h-9 items-center gap-2 rounded-lg border border-[var(--line-soft)] bg-[var(--glass-elevated)] px-3 text-sm font-semibold text-[var(--ink-muted)] shadow-sm">
            <Bot className="h-4 w-4 text-[var(--workspace-accent)]" aria-hidden="true" />
            Agent 후보 {totalAgentItems}개 · {loadedItemCount}/{visibleTotalCount}개 로드
          </span>
          {reviewView === "pending" ? <button
            type="button"
            data-testid="review-approve-loaded"
            onClick={() => openBulkConfirm("approve", loadedItemIds, "loaded")}
            disabled={Boolean(pendingAction) || loadedItemCount === 0}
            className="inline-flex h-9 items-center justify-center gap-2 rounded-lg border border-[#21132b] bg-[#21132b] px-3 text-sm font-semibold text-white shadow-sm disabled:cursor-not-allowed disabled:bg-neutral-400"
          >
            <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
            모두 승인
          </button> : null}
          {reviewView === "pending" ? <button
            type="button"
            data-testid="review-reject-loaded"
            onClick={() => openBulkConfirm("reject", loadedItemIds, "loaded")}
            disabled={Boolean(pendingAction) || loadedItemCount === 0}
            className="inline-flex h-9 items-center justify-center gap-2 rounded-lg border border-[var(--line-soft)] bg-[var(--glass-elevated)] px-3 text-sm font-semibold text-ink shadow-sm hover:bg-[var(--glass-strong)] disabled:cursor-not-allowed disabled:text-[var(--ink-muted)]"
          >
            <XCircle className="h-4 w-4" aria-hidden="true" />
            모두 반려
          </button> : null}
          <button
            type="button"
            onClick={() => void refreshReviewData(captureQueryContext())}
            disabled={invalidWorkflowContext || !isRenderedContextCurrent || visibleLoading}
            className="inline-flex h-9 items-center justify-center gap-2 rounded-lg border border-[var(--line-soft)] bg-[var(--glass-elevated)] px-3 text-sm font-semibold text-ink shadow-sm hover:bg-[var(--glass-strong)] disabled:cursor-not-allowed disabled:text-[var(--ink-muted)]"
          >
            <RefreshCw className="h-4 w-4" aria-hidden="true" />
            새로고침
          </button>
        </div>
      </div>

      {visibleError ? (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
          {authRequired ? (
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <p className="font-bold">로그인이 필요합니다.</p>
                <p className="mt-1">Slack 동기화로 생성된 검토 항목은 존재하지만, 현재 세션에서는 조회 권한을 확인할 수 없습니다.</p>
              </div>
              <Link
                href="/login"
                className="inline-flex h-9 items-center justify-center rounded-lg border border-red-300 bg-white px-3 text-sm font-semibold text-red-800 hover:bg-red-100"
              >
                로그인으로 이동
              </Link>
            </div>
          ) : (
            visibleError
          )}
        </div>
      ) : null}

      {isRenderedContextCurrent && promotionNotice ? (
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-900">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="font-bold">승인 완료</p>
              <p className="mt-1">
                {promotionNotice.itemTitle} 항목이 {itemTypeLabel(promotionNotice.result.target_type)} 지식으로 연결되었습니다.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              {promotionNotice.result.next_routes.map((route) => (
                <Link
                  key={route}
                  href={route}
                  className="inline-flex h-9 items-center rounded-lg border border-emerald-300 bg-white px-3 text-sm font-semibold text-emerald-900 hover:bg-emerald-100"
                >
                  {routeLabel(route)}
                </Link>
              ))}
            </div>
          </div>
        </div>
      ) : null}

      {reviewView === "pending" ? <section className="sticky top-24 z-10 rounded-xl border border-[var(--line-soft)] bg-[var(--glass-elevated)]/95 p-3 shadow-sm backdrop-blur">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            <button
              type="button"
              data-testid="review-select-all"
              aria-label={allLoadedSelected ? "로드된 검토 항목 선택 해제" : "로드된 검토 항목 전체 선택"}
              aria-pressed={allLoadedSelected}
              onClick={toggleAllLoadedSelection}
              disabled={loadedItemCount === 0}
              className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-[var(--line-soft)] bg-white text-[var(--ink)] shadow-sm hover:bg-[var(--glass-strong)] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {allLoadedSelected || someLoadedSelected ? <CheckSquare className="h-4 w-4" /> : <Square className="h-4 w-4" />}
            </button>
            <span data-testid="review-selected-count" className="text-sm font-bold text-[var(--ink)]">
              선택 {selectedLoadedIds.length}개
            </span>
            <select
              data-testid="review-bulk-project"
              value={bulkProjectState.value}
              onChange={(event) => {
                if (!bulkProjectState.disabled) setBulkProjectKey(event.target.value);
              }}
              disabled={bulkProjectState.disabled}
              className="h-9 min-w-[180px] rounded-lg border border-[var(--line-soft)] bg-white px-3 text-sm font-semibold text-[var(--ink)] outline-none focus:border-[#21132b]"
            >
              <option value="">프로젝트 선택</option>
              {bulkProjectState.projects.map((project) => (
                <option key={project.project_key} value={project.project_key}>
                  {project.name}
                </option>
              ))}
            </select>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              data-testid="review-bulk-approve"
              onClick={() => openBulkConfirm("approve", selectedLoadedIds, "selected")}
              disabled={Boolean(pendingAction) || selectedLoadedIds.length === 0}
              className="inline-flex h-9 items-center justify-center gap-2 rounded-lg border border-[#21132b] bg-[#21132b] px-3 text-sm font-semibold text-white shadow-sm disabled:cursor-not-allowed disabled:bg-neutral-400"
            >
              <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
              선택 승인
            </button>
            <button
              type="button"
              onClick={() => openBulkConfirm("reject", selectedLoadedIds, "selected")}
              disabled={Boolean(pendingAction) || selectedLoadedIds.length === 0}
              className="inline-flex h-9 items-center justify-center gap-2 rounded-lg border border-[var(--line-soft)] bg-white px-3 text-sm font-semibold text-ink shadow-sm hover:bg-[var(--glass-strong)] disabled:cursor-not-allowed disabled:text-[var(--ink-muted)]"
            >
              <XCircle className="h-4 w-4" aria-hidden="true" />
              선택 반려
            </button>
            <button
              type="button"
              onClick={() => openBulkConfirm("approve", duplicateItemIds, "similar")}
              disabled={Boolean(pendingAction) || duplicateItemIds.length === 0}
              className="hidden h-9 items-center justify-center gap-2 rounded-lg border border-[var(--line-soft)] bg-white px-3 text-sm font-semibold text-ink shadow-sm hover:bg-[var(--glass-strong)] disabled:cursor-not-allowed disabled:text-[var(--ink-muted)]"
            >
              <Layers className="h-4 w-4" aria-hidden="true" />
              중복/유사 승인
            </button>
            <button
              type="button"
              onClick={() => openBulkConfirm("reject", duplicateItemIds, "similar")}
              disabled={Boolean(pendingAction) || duplicateItemIds.length === 0}
              className="hidden h-9 items-center justify-center gap-2 rounded-lg border border-[var(--line-soft)] bg-white px-3 text-sm font-semibold text-ink shadow-sm hover:bg-[var(--glass-strong)] disabled:cursor-not-allowed disabled:text-[var(--ink-muted)]"
            >
              <XCircle className="h-4 w-4" aria-hidden="true" />
              중복/유사 반려
            </button>
          </div>
        </div>
      </section> : null}

      <section className="space-y-4">
        {visibleGroups.map((group) => {
          const isExpanded = expandedGroups[group.group_id];
          const hasMultiple = group.total_count > 1;
          const groupSelected = isGroupFullySelected(group);
          const groupPartiallySelected = isGroupPartiallySelected(group);

          return (
            <div key={group.group_id} className="group-container overflow-hidden rounded-xl border border-[var(--line-soft)] bg-[var(--glass-elevated)] shadow-sm">
              {/* Group Header */}
              <div 
                onClick={() => toggleGroup(group)}
                className="flex cursor-pointer items-center justify-between border-b border-[var(--line-soft)] bg-[var(--glass-strong)] px-4 py-3 hover:bg-[var(--glass-stronger)]"
              >
                <div className="flex items-center gap-3">
                  {reviewView === "pending" ? <button
                    type="button"
                    data-testid={`review-group-select-${group.group_id}`}
                    aria-label={`${group.title} 선택`}
                    aria-pressed={groupSelected}
                    onClick={(event) => toggleGroupSelection(event, group)}
                    className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-[var(--line-soft)] bg-white text-[var(--ink)] shadow-sm hover:bg-[var(--glass-strong)]"
                  >
                    {groupSelected || groupPartiallySelected ? <CheckSquare className="h-4 w-4" /> : <Square className="h-4 w-4" />}
                  </button> : null}
                  <span className="rounded-full border border-[var(--line-soft)] bg-white/50 px-2.5 py-0.5 text-xs font-bold text-[var(--ink-muted)]">
                    {itemTypeLabel(group.item_type)}
                  </span>
                  <h3 className="text-base font-bold">{group.title}</h3>
                  {hasMultiple && (
                    <span className="flex items-center gap-1 rounded-md bg-[var(--workspace-rail-active)] px-2 py-0.5 text-xs font-bold text-white">
                      <Layers className="h-3 w-3" />
                      {group.total_count}개 중복/유사
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-4">
                  {reviewView === "pending" && hasMultiple ? (
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        data-testid={`review-group-similar-approve-${group.group_id}`}
                        onClick={(event) => {
                          event.stopPropagation();
                          openBulkConfirm("approve", groupItemIds(group), "similar");
                        }}
                        disabled={Boolean(pendingAction)}
                        className="inline-flex h-8 items-center justify-center gap-1.5 rounded-lg border border-[#21132b] bg-[#21132b] px-2.5 text-xs font-semibold text-white shadow-sm disabled:cursor-not-allowed disabled:bg-neutral-400"
                      >
                        <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
                        중복/유사 승인
                      </button>
                      <button
                        type="button"
                        data-testid={`review-group-similar-reject-${group.group_id}`}
                        onClick={(event) => {
                          event.stopPropagation();
                          openBulkConfirm("reject", groupItemIds(group), "similar");
                        }}
                        disabled={Boolean(pendingAction)}
                        className="inline-flex h-8 items-center justify-center gap-1.5 rounded-lg border border-[var(--line-soft)] bg-white px-2.5 text-xs font-semibold text-ink shadow-sm hover:bg-[var(--glass-strong)] disabled:cursor-not-allowed disabled:text-[var(--ink-muted)]"
                      >
                        <XCircle className="h-3.5 w-3.5" aria-hidden="true" />
                        중복/유사 반려
                      </button>
                    </div>
                  ) : null}
                   <div className="text-right">
                      <p className="text-[10px] font-bold uppercase tracking-wider text-[var(--ink-muted)]">평균 신뢰도</p>
                      <p className="text-sm font-bold">{Math.round(group.avg_confidence * 100)}%</p>
                   </div>
                </div>
              </div>

              {/* Group Items */}
              {isExpanded && (
                <div className="divide-y divide-[var(--line-soft)] bg-white/30">
                  {group.items.map((item) => {
                    const isEditing = editingId === item.id;
                    const editPending = pendingAction === `${item.id}:edit`;
                    const agentName = stringField(item.payload.agent_name);
                    const isAgentItem = Boolean(agentName);
                    const promptVersion = isAgentItem ? reviewPromptLabel(item, agentName) : "";
                    const cost = isAgentItem ? reviewCostLabel(item, agentName) : "";
                    const preview = previews[item.id];
                    const projectSelectionRequired = needsProjectSelection(item, preview);
                    const canApprove = (preview?.can_approve ?? true) && !projectSelectionRequired;
                    const evidenceRows = item.source_evidence ?? [];
                    const isEvidenceRequestOpen = evidenceRequestId === item.id;
                    const evidenceRequestPending = pendingAction === `${item.id}:request-more-evidence`;
                    const workFields = mailDocsWorkFields(item);
                    const assignmentFields = projectAssignmentFields(item);
                    const isDeepLinked = deepLinkedItemId === item.id;

                    return (
                      <div
                        key={item.id}
                        id={`review-item-${item.id}`}
                        data-testid={`review-item-${item.id}`}
                        className={`scroll-mt-24 p-5 ${isDeepLinked ? "bg-white ring-2 ring-[var(--workspace-rail-active)] ring-inset" : ""}`}
                        onContextMenu={reviewView === "pending" ? (event) => openContextMenu(event, item) : undefined}
                      >
                        <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
                          {reviewView === "pending" ? <button
                            type="button"
                            data-testid={`review-select-${item.id}`}
                            aria-label={`${itemTitle(item)} 선택`}
                            aria-pressed={selectedItemIds.has(item.id)}
                            onClick={() => toggleItemSelection(item.id)}
                            className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-[var(--line-soft)] bg-white text-[var(--ink)] shadow-sm hover:bg-[var(--glass-strong)]"
                          >
                            {selectedItemIds.has(item.id) ? <CheckSquare className="h-4 w-4" /> : <Square className="h-4 w-4" />}
                          </button> : null}
                          <div className="min-w-0 flex-1">
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="rounded-full border border-[var(--line-soft)] bg-[var(--glass-strong)] px-2.5 py-1 text-xs font-semibold capitalize text-[var(--ink-muted)]">
                                {item.permission_level}
                              </span>
                              {isAgentItem ? (
                                <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-semibold ${agentBadgeClass(item, agentName)}`}>
                                  <Sparkles className="h-3 w-3" aria-hidden="true" />
                                  {agentBadgeLabel(item, agentName)}
                                </span>
                              ) : (
                                <span className="rounded-full bg-emerald-50 px-2.5 py-1 text-xs font-semibold text-emerald-700">
                                  Human/Connector
                                </span>
                              )}
                            </div>

                            {isEditing ? (
                              <div className="mt-3 max-w-3xl space-y-3">
                                <label className="block text-sm font-semibold">
                                  소속 프로젝트
                                  <select
                                    value={editProjectKey}
                                    onChange={(e) => setEditProjectKey(e.target.value)}
                                    className="mt-1 h-10 w-full rounded-lg border border-[var(--line-soft)] px-3 text-sm font-normal outline-none focus:border-[#21132b] bg-white"
                                  >
                                    <option value="">프로젝트 미지정</option>
                                    {definedProjects.map(p => (
                                      <option key={p.project_key} value={p.project_key}>{p.name}</option>
                                    ))}
                                  </select>
                                </label>
                                <label className="block text-sm font-semibold">
                                  제목
                                  <input
                                    value={editTitle}
                                    onChange={(event) => setEditTitle(event.target.value)}
                                    className="mt-1 h-10 w-full rounded-lg border border-[var(--line-soft)] px-3 text-sm font-normal outline-none focus:border-[#21132b]"
                                  />
                                </label>
                                <label className="block text-sm font-semibold">
                                  요약
                                  <textarea
                                    value={editSummary}
                                    onChange={(event) => setEditSummary(event.target.value)}
                                    rows={3}
                                    className="mt-1 w-full rounded-lg border border-[var(--line-soft)] px-3 py-2 text-sm font-normal leading-6 outline-none focus:border-[#21132b]"
                                  />
                                </label>
                                {mailDocsWorkFields(item) ? (
                                  <label className="block text-sm font-semibold">
                                    다음 행동
                                    <textarea
                                      value={editNextStep}
                                      onChange={(event) => setEditNextStep(event.target.value)}
                                      rows={2}
                                      className="mt-1 w-full rounded-lg border border-[var(--line-soft)] px-3 py-2 text-sm font-normal leading-6 outline-none focus:border-[#21132b]"
                                    />
                                  </label>
                                ) : null}
                              </div>
                            ) : (
                              <>
                                {workFields ? (
                                  <div className="mt-4 max-w-3xl rounded-lg border border-[var(--line-soft)] bg-white/70 p-4">
                                    <p className="text-[11px] font-bold uppercase tracking-wide text-[var(--ink-muted)]">
                                      Mail/Docs 업무 판단
                                    </p>
                                    {workFields.nextStep ? (
                                      <p className="mt-2 text-sm font-bold leading-6 text-[var(--ink)]">
                                        다음 행동: {workFields.nextStep}
                                      </p>
                                    ) : null}
                                    {workFields.taskSummary ? (
                                      <p className="mt-2 text-sm leading-6 text-[var(--ink)]">
                                        할 일: {workFields.taskSummary}
                                      </p>
                                    ) : null}
                                    {workFields.businessContext ? (
                                      <p className="mt-2 text-sm leading-6 text-[var(--ink-muted)]">
                                        맥락: {workFields.businessContext}
                                      </p>
                                    ) : null}
                                    <div className="mt-3 flex flex-wrap gap-2 text-xs font-semibold text-[var(--ink-muted)]">
                                      {workFields.assignee ? <span>담당자: {workFields.assignee}</span> : null}
                                      {workFields.dueDate ? <span>기한: {workFields.dueDate}</span> : null}
                                      {workFields.counterparty ? <span>상대: {workFields.counterparty}</span> : null}
                                    </div>
                                  </div>
                                ) : null}
                                {reviewView === "pending" ? <label className="mt-4 block max-w-sm text-sm font-semibold text-[var(--ink)]">
                                  프로젝트 지정
                                  <select
                                    aria-label="프로젝트 지정"
                                    value={stringField(item.payload.project_key)}
                                    onChange={(event) => void updateItemProject(item, event.target.value)}
                                    disabled={Boolean(pendingAction)}
                                    className="mt-1 h-10 w-full rounded-lg border border-[var(--line-soft)] bg-white px-3 text-sm font-normal outline-none focus:border-[#21132b] disabled:opacity-60"
                                  >
                                    <option value="">프로젝트 선택</option>
                                    {definedProjects.map((project) => (
                                      <option key={project.project_key} value={project.project_key}>
                                        {project.name}
                                      </option>
                                    ))}
                                  </select>
                                </label> : null}
                                {projectSelectionRequired ? (
                                  <div
                                    data-testid="project-selection-required"
                                    className="mt-3 max-w-3xl rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm font-semibold leading-6 text-amber-900"
                                  >
                                    <p>프로젝트 선택 후 승인할 수 있습니다.</p>
                                    <p className="mt-1 font-medium">
                                      등록된 프로젝트와 자동 매칭되지 않아 검토자 확인이 필요합니다.
                                      <Link href="/projects" className="ml-2 underline underline-offset-4">
                                        새 프로젝트 만들기
                                      </Link>
                                    </p>
                                  </div>
                                ) : null}
                                {assignmentFields ? (
                                  <div className="mt-4 max-w-3xl rounded-lg border border-[var(--line-soft)] bg-white/70 p-4">
                                    <p className="text-[11px] font-bold uppercase tracking-wide text-[var(--ink-muted)]">
                                      프로젝트 연결 후보
                                    </p>
                                    {assignmentFields.projectName ? (
                                      <p className="mt-2 text-sm font-bold leading-6 text-[var(--ink)]">
                                        추천 프로젝트: {assignmentFields.projectName}
                                      </p>
                                    ) : null}
                                    {assignmentFields.taskSummary ? (
                                      <p className="mt-2 text-sm leading-6 text-[var(--ink)]">
                                        연결 내용: {assignmentFields.taskSummary}
                                      </p>
                                    ) : null}
                                    {assignmentFields.evidenceReason ? (
                                      <p className="mt-2 text-sm leading-6 text-[var(--ink-muted)]">
                                        분류 근거: {assignmentFields.evidenceReason}
                                      </p>
                                    ) : null}
                                    <div className="mt-3 flex flex-wrap gap-2 text-xs font-semibold text-[var(--ink-muted)]">
                                      {assignmentFields.sourceTitle ? (
                                        <span>원본: {assignmentFields.sourceTitle}</span>
                                      ) : null}
                                      {assignmentFields.sourceType ? (
                                        <span>출처: {assignmentFields.sourceType}</span>
                                      ) : null}
                                    </div>
                                  </div>
                                ) : null}
                                {projectRoutingLabel(item) ? (
                                  <div className="mt-4 max-w-3xl rounded-lg border border-[var(--line-soft)] bg-white/70 p-4">
                                    <p className="text-[11px] font-bold uppercase tracking-wide text-[var(--ink-muted)]">
                                      {projectRoutingLabel(item)}
                                    </p>
                                    {projectRoutingSummary(item) ? (
                                      <p className="mt-2 text-sm font-bold leading-6 text-[var(--ink)]">
                                        {projectRoutingSummary(item)}
                                      </p>
                                    ) : null}
                                    {projectRoutingReason(item) ? (
                                      <p className="mt-2 text-sm leading-6 text-[var(--ink-muted)]">
                                        {projectRoutingReason(item)}
                                      </p>
                                    ) : null}
                                  </div>
                                ) : null}
                                <h4 className="mt-3 text-sm font-bold text-[var(--ink-muted)]">{reviewView === "auto" ? "상세 내용" : `#${item.id} 상세 내용`}</h4>
                                <p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--ink)]">
                                  {itemSummary(item)}
                                </p>
                              </>
                            )}

                            {isAgentItem ? (
                              <div className="mt-4 flex flex-wrap gap-3">
                                <MetadataTile label="Prompt" value={promptVersion} />
                                <MetadataTile label="Cost" value={cost} />
                              </div>
                            ) : null}

                            {reviewView === "pending" && preview ? (
                              <div className="mt-4 rounded-lg border border-[var(--line-soft)] bg-[var(--glass-strong)] p-3">
                                <div className="flex flex-wrap items-center justify-between gap-2">
                                  <p className="text-[10px] font-bold uppercase tracking-wide text-[var(--ink-muted)]">
                                    승인 시 저장 데이터 구조
                                  </p>
                                  <span
                                    className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${
                                      preview.can_approve
                                        ? "bg-emerald-100 text-emerald-800"
                                        : "bg-red-100 text-red-800"
                                    }`}
                                  >
                                    {preview.can_approve ? "승격 가능" : "필수 정보 부족"}
                                  </span>
                                </div>
                                <dl className="mt-2 grid gap-x-4 gap-y-2 sm:grid-cols-2">
                                  {Object.entries(preview.normalized_payload).map(([key, value]) => (
                                    <div key={key} className="min-w-0">
                                      <dt className="text-[10px] font-bold text-[var(--ink-muted)] uppercase">{key}</dt>
                                      <dd className="mt-0.5 truncate text-xs font-medium">{value || "-"}</dd>
                                    </div>
                                  ))}
                                </dl>
                              </div>
                            ) : null}
                          </div>

                          <div className="flex shrink-0 flex-col gap-3 sm:flex-row xl:items-start">
                            <div className="rounded-lg border border-[var(--line-soft)] bg-[var(--glass-strong)] px-3 py-2 text-center min-w-[80px]">
                              <p className="text-[10px] font-bold uppercase tracking-wide text-[var(--ink-muted)]">신뢰도</p>
                              <p className="mt-1 text-lg font-bold">{Math.round(item.confidence_score * 100)}%</p>
                            </div>
                            <SourceEvidenceDrawer
                              evidence={evidenceRows}
                              links={item.source_links}
                              snippets={item.source_snippets}
                              itemTitle={itemTitle(item)}
                              agentRunId={item.agent_run_id}
                            />
                          </div>
                        </div>

                        {reviewView === "pending" ? <div className="mt-4 flex flex-wrap gap-2 border-t border-dashed border-[var(--line-soft)] pt-4">
                          {isEditing ? (
                            <>
                              <button
                                type="button"
                                onClick={() => void saveEdit(item)}
                                disabled={Boolean(pendingAction)}
                                className="inline-flex h-9 items-center gap-2 rounded-lg border border-[#21132b] bg-[#21132b] px-3 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:bg-neutral-400"
                              >
                                <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
                                {editPending ? "저장 중" : "수정 저장"}
                              </button>
                              <button
                                type="button"
                                onClick={() => setEditingId(undefined)}
                                disabled={Boolean(pendingAction)}
                                className="inline-flex h-9 items-center gap-2 rounded-lg border border-[var(--line-soft)] bg-[var(--glass-elevated)] px-3 text-sm font-semibold text-ink hover:bg-[var(--glass-strong)] disabled:cursor-not-allowed disabled:text-[var(--ink-muted)]"
                              >
                                <XCircle className="h-4 w-4" aria-hidden="true" />
                                취소
                              </button>
                            </>
                          ) : (
                            <>
                              <button
                                type="button"
                                aria-label="승인"
                                onClick={() => void runStatusAction(item, "approve")}
                                disabled={Boolean(pendingAction) || !canApprove}
                                className="inline-flex h-9 items-center gap-2 rounded-lg border border-[#21132b] bg-[#21132b] px-3 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:bg-neutral-400"
                              >
                                <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
                                승인
                              </button>
                              <button
                                type="button"
                                onClick={() => void runStatusAction(item, "reject")}
                                disabled={Boolean(pendingAction)}
                                className="inline-flex h-9 items-center gap-2 rounded-lg border border-[var(--line-soft)] bg-[var(--glass-elevated)] px-3 text-sm font-semibold text-ink hover:bg-[var(--glass-strong)] disabled:cursor-not-allowed disabled:text-[var(--ink-muted)]"
                              >
                                <XCircle className="h-4 w-4" aria-hidden="true" />
                                반려
                              </button>
                              <button
                                type="button"
                                onClick={() => startEdit(item)}
                                disabled={Boolean(pendingAction)}
                                className="inline-flex h-9 items-center gap-2 rounded-lg border border-[var(--line-soft)] bg-[var(--glass-elevated)] px-3 text-sm font-semibold text-ink hover:bg-[var(--glass-strong)] disabled:cursor-not-allowed disabled:text-[var(--ink-muted)]"
                              >
                                <Pencil className="h-4 w-4" aria-hidden="true" />
                                수정
                              </button>
                              <button
                                type="button"
                                onClick={() => {
                                  setEvidenceRequestId(item.id);
                                  setEvidenceRequestNote(
                                    "현재 근거만으로는 승인하기 어렵습니다. 원문 링크, 담당자 발언, 결정 시점을 추가로 확인해주세요.",
                                  );
                                }}
                                disabled={Boolean(pendingAction)}
                                className="inline-flex h-9 items-center gap-2 rounded-lg border border-[var(--line-soft)] bg-[var(--glass-elevated)] px-3 text-sm font-semibold text-ink hover:bg-[var(--glass-strong)] disabled:cursor-not-allowed disabled:text-[var(--ink-muted)]"
                              >
                                <FileSearch className="h-4 w-4" aria-hidden="true" />
                                근거 추가 요청
                              </button>
                            </>
                          )}
                        </div> : <AutoReviewTrustPanel item={item} onChanged={() => refreshReviewData(captureQueryContext())} />}
                        {isEvidenceRequestOpen ? (
                          <div className="mt-3 rounded-lg border border-[var(--line-soft)] bg-[var(--glass-elevated)] p-3">
                            <label className="block text-sm font-semibold text-[var(--ink)]">
                              추가로 필요한 근거
                              <textarea
                                value={evidenceRequestNote}
                                onChange={(event) => setEvidenceRequestNote(event.target.value)}
                                rows={3}
                                className="mt-2 w-full rounded-lg border border-[var(--line-soft)] bg-[var(--surface)] px-3 py-2 text-sm font-normal leading-6 text-[var(--ink)] outline-none focus:border-[var(--workspace-rail-active)]"
                              />
                            </label>
                            <div className="mt-3 flex flex-wrap gap-2">
                              <button
                                type="button"
                                onClick={() =>
                                  void runStatusAction(item, "request-more-evidence", {
                                    note: evidenceRequestNote,
                                  })
                                }
                                disabled={Boolean(pendingAction)}
                                className="inline-flex h-9 items-center gap-2 rounded-lg border border-[#21132b] bg-[#21132b] px-3 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:bg-neutral-400"
                              >
                                <FileSearch className="h-4 w-4" aria-hidden="true" />
                                {evidenceRequestPending ? "요청 중" : "요청 보내기"}
                              </button>
                              <button
                                type="button"
                                onClick={() => {
                                  setEvidenceRequestId(undefined);
                                  setEvidenceRequestNote("");
                                }}
                                disabled={Boolean(pendingAction)}
                                className="inline-flex h-9 items-center gap-2 rounded-lg border border-[var(--line-soft)] bg-[var(--glass-elevated)] px-3 text-sm font-semibold text-ink hover:bg-[var(--glass-strong)] disabled:cursor-not-allowed disabled:text-[var(--ink-muted)]"
                              >
                                <XCircle className="h-4 w-4" aria-hidden="true" />
                                취소
                              </button>
                            </div>
                          </div>
                        ) : null}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}

        {!visibleLoading && !visibleError && visibleGroups.length === 0 ? (
          <div className="rounded-lg border border-[var(--line-soft)] bg-[var(--glass-elevated)] p-8 text-sm text-[var(--ink-muted)] shadow-sm text-center">
            {reviewView === "auto" ? "자동 승인된 항목이 없습니다." : "대기 중인 검토 항목이 없습니다."}
          </div>
        ) : null}
        {!visibleLoading && visibleHasMore ? (
          <div className="flex justify-center">
            <button
              type="button"
              onClick={() => void loadItems(loadedOffset + REVIEW_PAGE_SIZE, true)}
              className="inline-flex h-10 items-center justify-center rounded-lg border border-[var(--line-soft)] bg-[var(--glass-elevated)] px-4 text-sm font-semibold text-ink shadow-sm hover:bg-[var(--glass-strong)]"
            >
              더 보기
            </button>
          </div>
        ) : null}
        {visibleLoading && visibleGroups.length === 0 ? (
          <div className="rounded-lg border border-[var(--line-soft)] bg-[var(--glass-elevated)] p-8 text-sm text-[var(--ink-muted)] shadow-sm text-center">
            검토 항목을 불러오는 중입니다...
          </div>
        ) : null}
      </section>

      {typeof document !== "undefined" && isRenderedContextCurrent && contextMenu ? createPortal((
        <div
          data-testid="review-context-menu"
          className="fixed z-[110] w-60 overflow-hidden rounded-2xl border border-[var(--line-soft)] bg-white p-2 text-sm font-semibold text-[var(--ink)] shadow-2xl ring-1 ring-slate-950/5"
          style={{ left: contextMenu.x, top: contextMenu.y }}
          onClick={(event) => event.stopPropagation()}
        >
          <div className="border-b border-[var(--line-soft)] px-3 py-2">
            <p className="text-[11px] font-extrabold uppercase tracking-wide text-[var(--ink-muted)]">빠른 처리</p>
            <p className="mt-1 truncate text-sm font-extrabold text-[var(--ink)]">{itemTitle(contextMenu.item)}</p>
          </div>
          <button
            type="button"
            data-testid="review-context-approve"
            onClick={() => {
              const item = contextMenu.item;
              setContextMenu(undefined);
              void runStatusAction(item, "approve");
            }}
            className="mt-2 flex w-full items-center gap-2 rounded-xl px-3 py-2.5 text-left text-emerald-700 transition hover:bg-emerald-50"
          >
            <span className="grid h-7 w-7 place-items-center rounded-lg bg-emerald-100">
              <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
            </span>
            <span>
              <span className="block">승인</span>
              <span className="block text-[11px] font-semibold text-emerald-600">선택 항목을 지식 후보로 확정</span>
            </span>
          </button>
          <button
            type="button"
            data-testid="review-context-reject"
            onClick={() => {
              const item = contextMenu.item;
              setContextMenu(undefined);
              void runStatusAction(item, "reject");
            }}
            className="flex w-full items-center gap-2 rounded-xl px-3 py-2.5 text-left text-rose-700 transition hover:bg-rose-50"
          >
            <span className="grid h-7 w-7 place-items-center rounded-lg bg-rose-100">
              <XCircle className="h-4 w-4" aria-hidden="true" />
            </span>
            <span>
              <span className="block">반려</span>
              <span className="block text-[11px] font-semibold text-rose-600">큐에서 제외하고 감사 기록 남김</span>
            </span>
          </button>
        </div>
      ), document.body) : null}

      {typeof document !== "undefined" && isRenderedContextCurrent && bulkConfirm ? createPortal((
        <div data-testid="review-bulk-backdrop" className="fixed inset-0 z-[100] flex min-h-screen w-screen items-center justify-center bg-slate-950/45 px-4 backdrop-blur-sm">
          <div
            data-testid="review-bulk-confirm"
            className="w-full max-w-md rounded-2xl border border-white/70 bg-white p-5 shadow-2xl"
            role="dialog"
            aria-modal="true"
          >
            <div className="flex items-start gap-3">
              <span className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[var(--workspace-rail-active)] text-white">
                {bulkConfirm.action === "approve" ? <CheckCircle2 className="h-5 w-5" /> : <XCircle className="h-5 w-5" />}
              </span>
              <div className="min-w-0">
                <h2 className="text-base font-extrabold text-[var(--ink)]">
                  {bulkConfirm.scope === "loaded" ? "현재 로드된" : bulkConfirm.scope === "similar" ? "중복/유사" : "선택한"} 검토 항목 {bulkConfirm.itemIds.length}개를 모두{" "}
                  {bulkConfirm.action === "approve" ? "승인" : "반려"}할까요?
                </h2>
                <p className="mt-2 text-sm leading-6 text-[var(--ink-muted)]">
                  {bulkProjectKey ? "선택한 프로젝트를 먼저 반영한 뒤 처리합니다." : "프로젝트가 필요한 항목은 승인 단계에서 검증됩니다."}
                </p>
              </div>
            </div>
            <div className="mt-5 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setBulkConfirm(undefined)}
                disabled={Boolean(pendingAction)}
                className="inline-flex h-9 items-center justify-center rounded-lg border border-[var(--line-soft)] bg-white px-3 text-sm font-semibold text-[var(--ink)] hover:bg-[var(--glass-strong)] disabled:cursor-not-allowed disabled:opacity-60"
              >
                취소
              </button>
              <button
                type="button"
                data-testid="confirm-bulk-action"
                onClick={() => void executeBulkAction(bulkConfirm)}
                disabled={Boolean(pendingAction)}
                className="inline-flex h-9 items-center justify-center rounded-lg border border-[#21132b] bg-[#21132b] px-3 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:bg-neutral-400"
              >
                {pendingAction?.startsWith("bulk:") ? "처리 중" : "확인"}
              </button>
            </div>
          </div>
        </div>
      ), document.body) : null}
    </div>
  );
}

function MetadataTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-[var(--line-soft)] bg-[var(--glass-strong)] px-3 py-2">
      <p className="flex items-center gap-1 text-[11px] font-semibold uppercase tracking-wide text-[var(--ink-muted)]">
        <Coins className="h-3.5 w-3.5" aria-hidden="true" />
        {label}
      </p>
      <p className="mt-1 truncate text-sm font-semibold">{value}</p>
    </div>
  );
}

function mergeReviewGroups(current: ReviewGroup[], incoming: ReviewGroup[]) {
  const byId = new Map<string, ReviewGroup>();
  for (const group of current) {
    byId.set(group.group_id, { ...group, items: [...group.items] });
  }
  for (const group of incoming) {
    const existing = byId.get(group.group_id);
    if (!existing) {
      byId.set(group.group_id, { ...group, items: [...group.items] });
      continue;
    }
    const seenItemIds = new Set(existing.items.map((item) => item.id));
    const nextItems = [...existing.items];
    for (const item of group.items) {
      if (!seenItemIds.has(item.id)) {
        nextItems.push(item);
      }
    }
    byId.set(group.group_id, {
      ...existing,
      items: nextItems,
      total_count: nextItems.length,
      avg_confidence: nextItems.reduce((sum, item) => sum + item.confidence_score, 0) / Math.max(nextItems.length, 1),
    });
  }
  return Array.from(byId.values());
}
