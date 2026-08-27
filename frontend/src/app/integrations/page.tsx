"use client";

import {
  ArrowRight,
  CheckCircle2,
  ExternalLink,
  FileText,
  KeyRound,
  LockKeyhole,
  PlugZap,
  Radio,
  RefreshCw,
  ShieldCheck,
  Trash2,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { apiGet, apiPost, apiDelete } from "@/lib/api/client";
import {
  dryRunReviewWorkflow,
  getReviewWorkflowDiagnostic,
  launchReviewWorkflow,
  readReviewWorkflowError,
  ReviewWorkflowClientError,
} from "@/lib/api/reviewWorkflow";
import { notifyReviewQueueUpdated } from "@/lib/reviewQueueEvents";
import type {
  GoogleRuntimeStatus,
  IntegrationConnection,
  IntegrationManifest,
  IntegrationSyncResponse,
  OAuthInstallUrlResponse,
  SlackRuntimeStatus,
  DashboardResponse,
  ReviewWorkflowDiagnostic,
  ReviewWorkflowDryRun,
  ReviewWorkflowErrorCode,
  ReviewWorkflowLifecycleStatus,
  ReviewWorkflowSourceRef,
} from "@/lib/api/types";
import { ReviewCandidateLaunchPanel } from "./ReviewCandidateLaunchPanel";

const GOOGLE_CONNECTOR_TYPES = ["gmail", "drive", "calendar"] as const;
const SYNC_RUNNING_STAGES = [
  "원본 수집과 AI 분석을 진행 중입니다.",
  "프로젝트 분류와 검토 후보를 정리하고 있습니다.",
  "검토 항목을 저장하고 화면에 반영하고 있습니다.",
] as const;
const V2_SYNC_RUNNING_STAGES = [
  "원본을 수집하고 있습니다.",
  "변경 근거를 확인하고 있습니다.",
  "검토 후보 미리보기를 준비하고 있습니다.",
] as const;
const SYNC_STATUS_POLL_INTERVAL_MS = 1500;
const SYNC_STATUS_MAX_POLLS = 90;
const LOST_RESPONSE_RECOVERY_POLLS = 60;
const SYNC_BACKGROUND_NOTICE_DELAY_MS = 120_000;
const BACKGROUND_SYNC_CONTINUES_MESSAGE =
  "백그라운드에서 계속 진행 중입니다. 완료되면 작업 스트림의 최근 sync 상태에 반영됩니다.";

type SyncProgressState = {
  connectorType: string;
  displayName: string;
  status: "running" | "complete" | "error";
  stageIndex: number;
  progressPct: number;
  targetProgressPct: number;
  backgrounded: boolean;
  jobId?: string;
  lastMessage?: string;
  result?: IntegrationSyncResponse;
  errorMessage?: string;
  completionCopyMode: "legacy" | "v2_pending" | "v2_actionable" | "v2_unavailable";
  completionNotice?: string;
  reviewWorkflow?: {
    diagnostic: ReviewWorkflowDiagnostic;
    sourceRefs: ReviewWorkflowSourceRef[];
    batchKey: string;
    dryRun?: ReviewWorkflowDryRun;
    launchState:
      | "loading_preview"
      | "ready"
      | "launching"
      | "error"
      | "non_retryable_error"
      | "no_candidates"
      | "terminal";
    terminalStatus?: ReviewWorkflowLifecycleStatus;
    errorCode?: ReviewWorkflowErrorCode | null;
    errorMessage?: string;
  };
};

type IntegrationRuntimeStatus = SlackRuntimeStatus | GoogleRuntimeStatus;
type ConnectorLogo = () => ReactNode;

const DEFAULT_INTEGRATION_MANIFESTS: IntegrationManifest[] = [
  {
    type: "calendar",
    display_name: "Google Calendar",
    mode: "mock",
    status: "loading",
    auth_type: "oauth",
    required_scopes: [],
    sync_strategy: "incremental",
    cost_policy: "변경된 일정만 분석합니다.",
  },
  {
    type: "drive",
    display_name: "Google Drive",
    mode: "mock",
    status: "loading",
    auth_type: "oauth",
    required_scopes: [],
    sync_strategy: "incremental",
    cost_policy: "변경된 문서만 분석합니다.",
  },
  {
    type: "gmail",
    display_name: "Gmail",
    mode: "mock",
    status: "loading",
    auth_type: "oauth",
    required_scopes: [],
    sync_strategy: "incremental",
    cost_policy: "변경된 메일만 분석합니다.",
  },
  {
    type: "slack",
    display_name: "Slack",
    mode: "mock",
    status: "loading",
    auth_type: "oauth",
    required_scopes: [],
    sync_strategy: "incremental",
    cost_policy: "변경된 원천 데이터만 처리합니다.",
  },
];

/**
 * 연동 도구별 시각적 요소(아이콘, 색상, 설명 등) 정의
 */
const integrationVisuals = {
  slack: {
    logo: SlackLogo,
    description: "채널 메시지를 수집해 타임라인, 히스토리, 결정 후보를 만듭니다.",
  },
  gmail: {
    logo: GmailLogo,
    description: "메일 흐름을 요약하고 결정, 후속 작업, 히스토리 후보를 추출합니다.",
  },
  drive: {
    logo: GoogleDriveLogo,
    description: "사내 문서와 버전 정보를 회사 메모리의 근거로 연결합니다.",
  },
  calendar: {
    logo: GoogleCalendarLogo,
    description: "회의 일정과 시간 맥락을 히스토리 이벤트의 타임라인 근거로 사용합니다.",
  },
} satisfies Record<
  string,
  {
    logo: ConnectorLogo;
    description: string;
  }
>;

/**
 * 정의되지 않은 연동 도구에 대한 기본 시각적 설정
 */
const fallbackVisual = {
  logo: FallbackConnectorLogo,
  description: "공통 ingestion contract를 통해 회사 메모리로 연결됩니다.",
};

/**
 * 연동 관리 페이지 컴포넌트
 */
function sleep(ms: number) {
  return new Promise<void>((resolve) => window.setTimeout(resolve, ms));
}

function stageIndexFromProgress(progressPct?: number) {
  if (progressPct === undefined || progressPct < 50) {
    return 0;
  }
  if (progressPct < 90) {
    return 1;
  }
  return 2;
}

function nextDisplayedProgress(currentPct: number, targetPct: number) {
  const current = Math.max(0, Math.min(100, Math.round(currentPct)));
  const target = Math.max(0, Math.min(100, Math.round(targetPct)));
  if (current >= target) {
    return current;
  }
  return Math.min(target, current + 8);
}

function countFromSyncMessage(message: string | undefined, key: string) {
  const match = (message ?? "").match(new RegExp(`${key}=(\\d+)`));
  return match ? Number.parseInt(match[1], 10) : undefined;
}

function runtimeJobMatches(
  runtime: IntegrationRuntimeStatus,
  startedAtMs: number,
  jobId?: string,
) {
  const latest = runtime.latest_sync;
  if (!latest) {
    return false;
  }
  if (jobId) {
    return latest.job_id === jobId;
  }

  const timestamp = latest.updated_at ?? latest.created_at;
  if (!timestamp) {
    return false;
  }
  const timestampMs = new Date(timestamp).getTime();
  return Number.isFinite(timestampMs) && timestampMs >= startedAtMs - 5000;
}

function shouldRecoverLostSyncResponse(message: string) {
  return /internal server error|socket|network|failed to fetch|request failed with 5/i.test(message);
}

function isBackgroundSyncContinuation(message: string) {
  return message.includes("백그라운드에서 계속 진행 중입니다");
}

function reviewBatchKey(jobId: string, sourceRefs: ReviewWorkflowSourceRef[]) {
  return `${jobId}:${sourceRefs
    .map((source) => `${source.source_type}:${source.source_id}:${source.version_or_signature}`)
    .join("|")}`;
}

function reviewCompletionCopyMode(
  connectorType: string,
  diagnostic?: ReviewWorkflowDiagnostic,
): SyncProgressState["completionCopyMode"] {
  if (connectorType === "slack" || (diagnostic && (!diagnostic.enabled || !diagnostic.available))) {
    return "legacy";
  }
  return "v2_pending";
}

const NON_RETRYABLE_LAUNCH_ERROR_CODES = new Set<ReviewWorkflowErrorCode>([
  "evidence_changed",
  "idempotency_key_reused",
  "budget_exceeded",
]);

function syncResponseFromRuntimeStatus(
  connectorType: string,
  runtime: IntegrationRuntimeStatus,
  fallback?: IntegrationSyncResponse,
): IntegrationSyncResponse | undefined {
  const latest = runtime.latest_sync;
  if (!latest) {
    return undefined;
  }

  const message = latest.message ?? "";
  const slackSummary =
    runtime.connector_type === "slack" ? runtime.latest_sync_summary : undefined;
  const pendingReviewCount =
    runtime.connector_type === "slack"
      ? (countFromSyncMessage(message, "pending_review_items") ?? runtime.agent_bridge.pending_review_count)
      : (countFromSyncMessage(message, "pending_review_items") ?? fallback?.pending_review_count ?? 0);

  return {
    job_id: latest.job_id,
    connector_type: connectorType,
    status: latest.status,
    created_review_items:
      slackSummary?.created_review_items ??
      countFromSyncMessage(message, "created_review_items") ??
      fallback?.created_review_items ??
      0,
    pending_review_count: pendingReviewCount,
    fetched_events:
      slackSummary?.fetched_events ??
      countFromSyncMessage(message, "fetched") ??
      fallback?.fetched_events ??
      0,
    skipped_events:
      slackSummary?.skipped_events ??
      countFromSyncMessage(message, "skipped_events") ??
      fallback?.skipped_events ??
      0,
    parser_status_counts: fallback?.parser_status_counts ?? {},
    changed_source_ids: latest.changed_source_ids ?? fallback?.changed_source_ids ?? [],
    changed_source_refs: latest.changed_source_refs ?? fallback?.changed_source_refs ?? [],
    agent_generated_items: fallback?.agent_generated_items ?? 0,
    project_assignment_items: fallback?.project_assignment_items ?? 0,
  };
}

export default function IntegrationsPage() {
  const [manifests, setManifests] = useState<IntegrationManifest[]>([]);
  const [connections, setConnections] = useState<IntegrationConnection[]>([]);
  const [, setSlackRuntime] = useState<SlackRuntimeStatus>();
  const [selectedSlackChannels, setSelectedSlackChannels] = useState<string[]>([]);
  const [, setGoogleRuntimeByType] = useState<Record<string, GoogleRuntimeStatus>>({});
  const [dashboardSummary, setDashboardSummary] = useState<DashboardResponse>();
  const [slackOAuth, setSlackOAuth] = useState<OAuthInstallUrlResponse>();
  const [googleOAuthByType, setGoogleOAuthByType] = useState<Record<string, OAuthInstallUrlResponse>>({});
  const [pendingType, setPendingType] = useState<string>();
  const [syncProgress, setSyncProgress] = useState<SyncProgressState>();
  const [syncModalOpen, setSyncModalOpen] = useState(false);
  const [reviewWorkflowDiagnostic, setReviewWorkflowDiagnostic] = useState<ReviewWorkflowDiagnostic>();
  const activeSyncJobRef = useRef<string | undefined>(undefined);
  const activeReviewBatchRef = useRef<string | undefined>(undefined);

  // 초기 로드 시 다양한 연동 정보 및 상태 조회
  useEffect(() => {
    let active = true;
    
    // 연동 가능한 커넥터 목록(Manifest) 조회
    apiGet<IntegrationManifest[]>("/api/v1/integrations")
      .then((manifestResult) => {
        if (active) {
          setManifests(manifestResult);
        }
      })
      .catch(() => {
        if (active) {
          setManifests([]);
        }
      });

    // 현재 활성화된 연결(Credentials/Token 상태 등) 조회
    apiGet<IntegrationConnection[]>("/api/v1/integrations/connections")
      .then((connectionResult) => {
        if (active) {
          setConnections(connectionResult);
        }
      })
      .catch(() => {
        if (active) {
          setConnections([]);
        }
      });

    // 대시보드 요약 정보(소스별 카운트 등) 조회
    apiGet<DashboardResponse>("/api/v1/dashboard")
      .then((summary) => {
        if (active) {
          setDashboardSummary(summary);
        }
      })
      .catch(() => {
        if (active) {
          setDashboardSummary(undefined);
        }
      });

    getReviewWorkflowDiagnostic()
      .then((diagnostic) => {
        if (active) {
          setReviewWorkflowDiagnostic(diagnostic);
        }
      })
      .catch(() => {
        if (active) {
          setReviewWorkflowDiagnostic(undefined);
        }
      });

    // Slack OAuth 설치 URL 조회
    apiGet<OAuthInstallUrlResponse>("/api/v1/integrations/slack/oauth/install-url")
      .then((slackOAuthResult) => {
        if (active) {
          setSlackOAuth(slackOAuthResult);
        }
      })
      .catch(() => {
        if (active) {
          setSlackOAuth({
            connector_type: "slack",
            configured: false,
            install_url: null,
            state: null,
            required_scopes: [],
          });
        }
      });

    // Slack 운영 상태(채널 선택 등) 조회
    apiGet<SlackRuntimeStatus>("/api/v1/integrations/slack/runtime-status")
      .then((status) => {
        if (active) {
          setSlackRuntime(status);
          setSelectedSlackChannels(status.selected_channel_ids);
        }
      })
      .catch(() => {
        if (active) {
          setSlackRuntime(undefined);
        }
      });

    // Google 커넥터들(Gmail, Drive, Calendar)의 OAuth 및 상태 조회
    GOOGLE_CONNECTOR_TYPES.forEach((connectorType) => {
      apiGet<OAuthInstallUrlResponse>(`/api/v1/integrations/${connectorType}/oauth/install-url`)
        .then((googleOAuthResult) => {
          if (active) {
            setGoogleOAuthByType((current) => ({
              ...current,
              [connectorType]: googleOAuthResult,
            }));
          }
        })
        .catch(() => {
          if (active) {
            setGoogleOAuthByType((current) => ({
              ...current,
              [connectorType]: {
                connector_type: connectorType,
                configured: false,
                install_url: null,
                state: null,
                required_scopes: [],
              },
            }));
          }
        });

      apiGet<GoogleRuntimeStatus>(`/api/v1/integrations/${connectorType}/runtime-status`)
        .then((runtimeStatus) => {
          if (active) {
            setGoogleRuntimeByType((current) => ({
              ...current,
              [connectorType]: runtimeStatus,
            }));
          }
        })
        .catch(() => {
          if (active) {
            setGoogleRuntimeByType((current) => {
              const next = { ...current };
              delete next[connectorType];
              return next;
            });
          }
        });
    });

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!reviewWorkflowDiagnostic) {
      return;
    }
    setSyncProgress((current) => {
      if (!current || current.connectorType === "slack") {
        return current;
      }
      const completionCopyMode = reviewCompletionCopyMode(
        current.connectorType,
        reviewWorkflowDiagnostic,
      );
      return current.completionCopyMode === completionCopyMode
        ? current
        : { ...current, completionCopyMode };
    });
  }, [reviewWorkflowDiagnostic]);

  useEffect(() => {
    if (syncProgress?.status !== "running") {
      return undefined;
    }

    const timer = window.setInterval(() => {
      setSyncProgress((current) => {
        if (!current || current.status !== "running") {
          return current;
        }
        return {
          ...current,
          stageIndex: Math.min(current.stageIndex + 1, SYNC_RUNNING_STAGES.length - 1),
        };
      });
    }, 7000);

    return () => window.clearInterval(timer);
  }, [syncProgress?.connectorType, syncProgress?.status]);

  const displayedProgressPct = syncProgress?.progressPct;
  const targetProgressPct = syncProgress?.targetProgressPct;

  useEffect(() => {
    if (syncProgress?.status !== "running") {
      return undefined;
    }
    if (displayedProgressPct === undefined || targetProgressPct === undefined) {
      return undefined;
    }
    if (displayedProgressPct >= targetProgressPct) {
      return undefined;
    }

    const timer = window.setInterval(() => {
      setSyncProgress((current) => {
        if (!current || current.status !== "running") {
          return current;
        }
        return {
          ...current,
          progressPct: nextDisplayedProgress(current.progressPct, current.targetProgressPct),
        };
      });
    }, 500);

    return () => window.clearInterval(timer);
  }, [
    syncProgress?.connectorType,
    syncProgress?.status,
    displayedProgressPct,
    targetProgressPct,
  ]);


  async function refreshDashboardSummary() {
    try {
      const summary = await apiGet<DashboardResponse>("/api/v1/dashboard");
      setDashboardSummary(summary);
      notifyReviewQueueUpdated();
    } catch {
      setDashboardSummary(undefined);
    }
  }

  async function refreshRuntimeAfterMutation(type: string) {
    if (type === "slack") {
      try {
        const status = await apiGet<SlackRuntimeStatus>("/api/v1/integrations/slack/runtime-status");
        setSlackRuntime(status);
      } catch {
        setSlackRuntime(undefined);
      }
    }

    if (GOOGLE_CONNECTOR_TYPES.includes(type as (typeof GOOGLE_CONNECTOR_TYPES)[number])) {
      try {
        const status = await apiGet<GoogleRuntimeStatus>(`/api/v1/integrations/${type}/runtime-status`);
        setGoogleRuntimeByType((current) => ({ ...current, [type]: status }));
      } catch {
        setGoogleRuntimeByType((current) => {
          const next = { ...current };
          delete next[type];
          return next;
        });
      }
    }

    await refreshDashboardSummary();
  }

  async function loadRuntimeStatus(type: string): Promise<IntegrationRuntimeStatus | undefined> {
    if (type === "slack") {
      const status = await apiGet<SlackRuntimeStatus>("/api/v1/integrations/slack/runtime-status");
      setSlackRuntime(status);
      return status;
    }

    if (GOOGLE_CONNECTOR_TYPES.includes(type as (typeof GOOGLE_CONNECTOR_TYPES)[number])) {
      const status = await apiGet<GoogleRuntimeStatus>(`/api/v1/integrations/${type}/runtime-status`);
      setGoogleRuntimeByType((current) => ({ ...current, [type]: status }));
      return status;
    }

    return undefined;
  }

  async function waitForSyncCompletion(
    type: string,
    jobId: string,
    fallback: IntegrationSyncResponse,
  ): Promise<IntegrationSyncResponse> {
    for (let attempt = 0; attempt < SYNC_STATUS_MAX_POLLS; attempt += 1) {
      const runtime = await loadRuntimeStatus(type).catch(() => undefined);
      if (runtime?.latest_sync && runtimeJobMatches(runtime, Date.now(), jobId)) {
        const latest = runtime.latest_sync;
        setSyncProgress((current) =>
          current?.connectorType === type && current.status === "running"
            ? {
                ...current,
                stageIndex: stageIndexFromProgress(latest.progress_pct),
                targetProgressPct: latest.progress_pct,
                jobId: latest.job_id,
                lastMessage: latest.message,
              }
            : current,
        );

        if (latest.status === "failed") {
          throw new Error(latest.message || "동기화 작업이 실패했습니다.");
        }
        if (latest.status === "complete") {
          return syncResponseFromRuntimeStatus(type, runtime, fallback) ?? fallback;
        }
      }
      await sleep(SYNC_STATUS_POLL_INTERVAL_MS);
    }

    throw new Error("동기화 작업이 백그라운드에서 계속 진행 중입니다. 잠시 후 작업 스트림을 확인해 주세요.");
  }

  async function recoverCompletedSyncAfterLostResponse(
    type: string,
    startedAtMs: number,
  ): Promise<IntegrationSyncResponse | undefined> {
    for (let attempt = 0; attempt < LOST_RESPONSE_RECOVERY_POLLS; attempt += 1) {
      const runtime = await loadRuntimeStatus(type).catch(() => undefined);
      if (runtime?.latest_sync && runtimeJobMatches(runtime, startedAtMs)) {
        const latest = runtime.latest_sync;
        setSyncProgress((current) =>
          current?.connectorType === type && current.status === "running"
            ? {
                ...current,
                stageIndex: stageIndexFromProgress(latest.progress_pct),
                targetProgressPct: latest.progress_pct,
                jobId: latest.job_id,
                lastMessage: latest.message,
              }
            : current,
        );

        if (latest.status === "failed") {
          throw new Error(latest.message || "동기화 작업이 실패했습니다.");
        }
        if (latest.status === "complete") {
          return syncResponseFromRuntimeStatus(type, runtime);
        }
      }
      await sleep(SYNC_STATUS_POLL_INTERVAL_MS);
    }

    return undefined;
  }

  async function loadReviewCandidatePreview({
    type,
    jobId,
    batchKey,
    diagnostic,
    sourceRefs,
  }: {
    type: string;
    jobId: string;
    batchKey: string;
    diagnostic: ReviewWorkflowDiagnostic;
    sourceRefs: ReviewWorkflowSourceRef[];
  }) {
    if (activeSyncJobRef.current !== jobId || activeReviewBatchRef.current !== batchKey) {
      return;
    }
    setSyncProgress((current) =>
      current?.connectorType === type &&
      current.jobId === jobId &&
      current.status === "complete" &&
      current.reviewWorkflow?.batchKey === batchKey
        ? {
            ...current,
            reviewWorkflow: {
              ...current.reviewWorkflow,
              dryRun: undefined,
              launchState: "loading_preview",
              errorMessage: undefined,
            },
          }
        : current,
    );

    try {
      const dryRun = await dryRunReviewWorkflow({
        source_refs: sourceRefs,
        agent_names: diagnostic.default_agent_names,
      });
      if (activeSyncJobRef.current !== jobId || activeReviewBatchRef.current !== batchKey) {
        return;
      }
      setSyncProgress((current) =>
        current?.connectorType === type &&
        current.jobId === jobId &&
        current.status === "complete" &&
        current.reviewWorkflow?.batchKey === batchKey
          ? {
              ...current,
              reviewWorkflow: { ...current.reviewWorkflow, dryRun, launchState: "ready" },
            }
          : current,
      );
    } catch {
      if (activeSyncJobRef.current !== jobId || activeReviewBatchRef.current !== batchKey) {
        return;
      }
      setSyncProgress((current) =>
        current?.connectorType === type &&
        current.jobId === jobId &&
        current.status === "complete" &&
        current.reviewWorkflow?.batchKey === batchKey
          ? {
              ...current,
              reviewWorkflow: {
                ...current.reviewWorkflow,
                dryRun: undefined,
                launchState: "error",
                errorMessage: "비용을 확인하지 못했습니다.",
              },
            }
          : current,
      );
    }
  }

  async function prepareReviewCandidateLaunch(type: string, result: IntegrationSyncResponse) {
    const sourceRefs = result.changed_source_refs ?? [];
    let diagnostic = reviewWorkflowDiagnostic;
    let diagnosticLoadFailed = false;
    if (!diagnostic) {
      try {
        diagnostic = await getReviewWorkflowDiagnostic();
        setReviewWorkflowDiagnostic(diagnostic);
      } catch {
        diagnosticLoadFailed = true;
      }
    }
    setSyncProgress((current) =>
      current?.connectorType === type && current.jobId === result.job_id
        ? {
            ...current,
            completionCopyMode: diagnosticLoadFailed && type !== "slack"
              ? "v2_unavailable"
              : reviewCompletionCopyMode(type, diagnostic),
            completionNotice: diagnosticLoadFailed && type !== "slack"
              ? "검토 후보 기능을 확인할 수 없습니다. 데이터 변경 후 다시 동기화해 주세요."
              : undefined,
          }
        : current,
    );
    if (
      activeSyncJobRef.current !== result.job_id ||
      type === "slack" ||
      diagnosticLoadFailed ||
      !diagnostic?.enabled ||
      !diagnostic.available
    ) {
      return;
    }

    if (sourceRefs.length === 0) {
      setSyncProgress((current) =>
        current?.connectorType === type && current.jobId === result.job_id
          ? {
              ...current,
              completionCopyMode: "v2_unavailable",
              completionNotice: "이번 동기화에서 변경 근거를 찾지 못했습니다. 검토 후보를 만들 수 없습니다.",
            }
          : current,
      );
      return;
    }

    const batchKey = reviewBatchKey(result.job_id, sourceRefs);
    activeReviewBatchRef.current = batchKey;
    setSyncProgress((current) =>
      current?.connectorType === type && current.jobId === result.job_id && current.status === "complete"
          ? {
            ...current,
            completionCopyMode: "v2_actionable",
            completionNotice: undefined,
            reviewWorkflow: {
              diagnostic,
              sourceRefs,
              batchKey,
              launchState: "loading_preview",
            },
          }
        : current,
    );
    await loadReviewCandidatePreview({
      type,
      jobId: result.job_id,
      batchKey,
      diagnostic,
      sourceRefs,
    });
  }

  async function markSyncComplete(type: string, result: IntegrationSyncResponse) {
    if (activeSyncJobRef.current !== result.job_id) {
      return;
    }
    setSyncProgress((current) =>
      current?.connectorType === type
        ? {
            ...current,
            status: "complete",
            stageIndex: SYNC_RUNNING_STAGES.length - 1,
            progressPct: 100,
            targetProgressPct: 100,
            jobId: result.job_id,
            result,
          }
        : current,
    );
    await refreshRuntimeAfterMutation(type);
    await prepareReviewCandidateLaunch(type, result);
  }

  async function launchReviewCandidates() {
    const reviewWorkflow = syncProgress?.reviewWorkflow;
    const jobId = syncProgress?.jobId;
    const batchKey = reviewWorkflow?.batchKey;
    if (
      !syncProgress ||
      !reviewWorkflow ||
      !jobId ||
      !batchKey ||
      !reviewWorkflow.dryRun ||
      reviewWorkflow.launchState === "launching" ||
      activeSyncJobRef.current !== jobId ||
      activeReviewBatchRef.current !== batchKey
    ) {
      return;
    }

    const clientRequestId = `review:${jobId}:${reviewWorkflow.dryRun.selection_policy_version}`;
    setSyncProgress((current) =>
      current?.jobId === jobId && current.reviewWorkflow?.batchKey === batchKey
        ? {
            ...current,
            reviewWorkflow: { ...current.reviewWorkflow, launchState: "launching", errorMessage: undefined },
          }
        : current,
    );

    try {
      const status = await launchReviewWorkflow({
        source_refs: reviewWorkflow.sourceRefs,
        agent_names: reviewWorkflow.diagnostic.default_agent_names,
        client_request_id: clientRequestId,
      });
      if (activeSyncJobRef.current !== jobId || activeReviewBatchRef.current !== batchKey) {
        return;
      }
      if (status.status === "awaiting_human_review") {
        window.location.assign(`/review?workflow_thread_id=${encodeURIComponent(status.thread_id)}`);
        return;
      }
      setSyncProgress((current) =>
        current?.jobId === jobId && current.reviewWorkflow?.batchKey === batchKey
          ? {
              ...current,
              reviewWorkflow: {
                ...current.reviewWorkflow,
                launchState:
                  status.status === "completed" && status.review_item_count === 0
                    ? "no_candidates"
                    : "terminal",
                terminalStatus: status.status,
              },
            }
          : current,
      );
    } catch (error) {
      const safeError =
        error instanceof ReviewWorkflowClientError
          ? error
          : readReviewWorkflowError(error);
      const nonRetryable =
        safeError.code !== null && NON_RETRYABLE_LAUNCH_ERROR_CODES.has(safeError.code);
      if (activeSyncJobRef.current !== jobId || activeReviewBatchRef.current !== batchKey) {
        return;
      }
      setSyncProgress((current) =>
        current?.jobId === jobId && current.reviewWorkflow?.batchKey === batchKey
          ? {
              ...current,
              reviewWorkflow: {
                ...current.reviewWorkflow,
                launchState: nonRetryable ? "non_retryable_error" : "error",
                errorCode: safeError.code,
                errorMessage: nonRetryable ? undefined : safeError.message,
              },
            }
          : current,
      );
    }
  }

  async function retryReviewCandidatePreview() {
    const reviewWorkflow = syncProgress?.reviewWorkflow;
    const jobId = syncProgress?.jobId;
    if (
      !syncProgress ||
      !reviewWorkflow ||
      !jobId ||
      reviewWorkflow.launchState !== "error" ||
      activeSyncJobRef.current !== jobId ||
      activeReviewBatchRef.current !== reviewWorkflow.batchKey
    ) {
      return;
    }
    await loadReviewCandidatePreview({
      type: syncProgress.connectorType,
      jobId,
      batchKey: reviewWorkflow.batchKey,
      diagnostic: reviewWorkflow.diagnostic,
      sourceRefs: reviewWorkflow.sourceRefs,
    });
  }

  const visibleManifests = useMemo(
    () => (manifests.length > 0 ? manifests : DEFAULT_INTEGRATION_MANIFESTS),
    [manifests],
  );

  /**
   * 특정 커넥터의 데이터 동기화(Sync)를 시작합니다.
   */
  async function startSync(type: string) {
    const displayName = connectorDisplayName(type, manifests);
    const startedAtMs = Date.now();
    activeSyncJobRef.current = undefined;
    activeReviewBatchRef.current = undefined;
    setPendingType(type);
    setSyncProgress({
      connectorType: type,
      displayName,
      status: "running",
      stageIndex: 0,
      progressPct: 0,
      targetProgressPct: 10,
      backgrounded: false,
      completionCopyMode: reviewCompletionCopyMode(type, reviewWorkflowDiagnostic),
    });
    setSyncModalOpen(true);
    const backgroundNoticeTimer = window.setTimeout(() => {
      setSyncProgress((current) =>
        current?.connectorType === type && current.status === "running"
          ? {
              ...current,
              backgrounded: true,
              errorMessage: BACKGROUND_SYNC_CONTINUES_MESSAGE,
            }
          : current,
      );
    }, SYNC_BACKGROUND_NOTICE_DELAY_MS);

    try {
      const result = await apiPost<IntegrationSyncResponse>(
        `/api/v1/integrations/${type}/sync`,
        type === "slack"
          ? { selected_channel_ids: selectedSlackChannels, run_async: true }
          : { run_async: true },
      );
      activeSyncJobRef.current = result.job_id;
      setSyncProgress((current) =>
        current?.connectorType === type
          ? {
              ...current,
              jobId: result.job_id,
              progressPct: result.status === "complete" ? 100 : current.progressPct,
              targetProgressPct: result.status === "complete" ? 100 : Math.max(current.targetProgressPct, 10),
            }
          : current,
      );
      if (result.status === "failed") {
        throw new Error("동기화 작업이 실패했습니다.");
      }
      const completedResult =
        result.status === "complete"
          ? result
          : await waitForSyncCompletion(type, result.job_id, result);
      await markSyncComplete(type, completedResult);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "동기화에 실패했습니다.";
      if (shouldRecoverLostSyncResponse(message)) {
        try {
          const recovered = await recoverCompletedSyncAfterLostResponse(type, startedAtMs);
          if (recovered) {
            activeSyncJobRef.current = recovered.job_id;
            await markSyncComplete(type, recovered);
            return;
          }
        } catch (recoveryError) {
          const recoveryMessage =
            recoveryError instanceof Error
              ? recoveryError.message
              : "동기화 작업 상태 확인에 실패했습니다.";
          setSyncProgress((current) =>
            current?.connectorType === type
              ? {
                  ...current,
                  status: "error",
                  targetProgressPct: current.progressPct,
                  errorMessage: recoveryMessage,
                }
              : current,
          );
          return;
        }
      }
      if (isBackgroundSyncContinuation(message)) {
        setSyncProgress((current) =>
          current?.connectorType === type
            ? {
                ...current,
                status: "running",
                backgrounded: true,
                targetProgressPct: Math.max(current.targetProgressPct, 10),
                errorMessage: BACKGROUND_SYNC_CONTINUES_MESSAGE,
              }
            : current,
        );
        return;
      }
      setSyncProgress((current) =>
        current?.connectorType === type
          ? {
              ...current,
              status: "error",
              targetProgressPct: current.progressPct,
              errorMessage: message,
            }
          : current,
      );
    } finally {
      window.clearTimeout(backgroundNoticeTimer);
      setPendingType(undefined);
    }
  }

  async function refreshBackgroundSyncProgress(type: string) {
    const runtime = await loadRuntimeStatus(type).catch(() => undefined);
    const latest = runtime?.latest_sync;
    if (!runtime || !latest) return;
    if (syncProgress?.jobId && latest.job_id !== syncProgress.jobId) return;

    if (latest.status === "complete") {
      const result = syncResponseFromRuntimeStatus(type, runtime, syncProgress?.result);
      if (result) await markSyncComplete(type, result);
      return;
    }

    if (latest.status === "failed") {
      setSyncProgress((current) =>
        current?.connectorType === type
          ? {
              ...current,
              status: "error",
              progressPct: 100,
              targetProgressPct: 100,
              stageIndex: SYNC_RUNNING_STAGES.length - 1,
              lastMessage: latest.message,
              errorMessage: latest.message || "동기화 작업이 실패했습니다.",
            }
          : current,
      );
      return;
    }

    setSyncProgress((current) =>
      current?.connectorType === type
        ? {
            ...current,
            status: "running",
            targetProgressPct: latest.progress_pct,
            stageIndex: stageIndexFromProgress(latest.progress_pct),
            jobId: latest.job_id,
            lastMessage: latest.message,
          }
        : current,
    );
  }

  useEffect(() => {
    if (!syncProgress || syncProgress.status !== "running" || !syncProgress.backgrounded) {
      return;
    }

    const intervalId = window.setInterval(() => {
      void refreshBackgroundSyncProgress(syncProgress.connectorType);
    }, SYNC_STATUS_POLL_INTERVAL_MS);

    return () => window.clearInterval(intervalId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [syncProgress?.connectorType, syncProgress?.status, syncProgress?.backgrounded]);

  /**
   * 연동 해제(Disconnect)를 실행합니다.
   */
  async function disconnect(type: string) {
    if (!confirm(`${type} 연동을 해제하시겠습니까? 관련 자격 증명이 삭제됩니다.`)) {
      return;
    }

    try {
      await apiDelete(`/api/v1/integrations/${type}`);
      
      // 연결 목록 갱신
      const connectionResult = await apiGet<IntegrationConnection[]>("/api/v1/integrations/connections");
      setConnections(connectionResult);
      
      // 구글 런타임 상태 갱신
      if (GOOGLE_CONNECTOR_TYPES.includes(type as (typeof GOOGLE_CONNECTOR_TYPES)[number])) {
        setGoogleRuntimeByType((current) => {
          const next = { ...current };
          delete next[type];
          return next;
        });
      }
      
      // 슬랙 런타임 상태 갱신
      if (type === "slack") {
        setSlackRuntime(undefined);
      }
    } catch (caught) {
      window.alert(caught instanceof Error ? caught.message : "연동 해제에 실패했습니다.");
    }
  }

  async function startOAuth(displayName: string, oauth?: OAuthInstallUrlResponse) {
    if (!oauth?.install_url) {
      window.alert(`${displayName} OAuth 설정이 아직 준비되지 않았습니다. .env의 client id와 redirect URI를 확인하세요.`);
      return;
    }

    if (oauth.install_url === "__direct_connect__") {
      try {
        const connection = await apiPost<IntegrationConnection>("/api/v1/integrations/slack/direct-connect");
        setConnections((current) => {
          const filtered = current.filter((item) => item.connector_type !== "slack");
          return [...filtered, connection];
        });
        
        // 연결 성공 후 런타임 상태 갱신
        apiGet<SlackRuntimeStatus>("/api/v1/integrations/slack/runtime-status")
          .then((status) => {
            setSlackRuntime(status);
            setSelectedSlackChannels(status.selected_channel_ids);
          })
          .catch(() => undefined);
          
      } catch (caught) {
        window.alert(caught instanceof Error ? caught.message : "Slack 직접 연결에 실패했습니다.");
      }
      return;
    }

    window.location.assign(oauth.install_url);
  }

  return (
    <div className="reference-dashboard utility-workspace space-y-5">
      {syncProgress && syncModalOpen ? (
        <SyncProgressModal
          progress={syncProgress}
          onBackground={() => {
            setSyncProgress((current) => (current ? { ...current, backgrounded: true } : current));
            setSyncModalOpen(false);
          }}
          onClose={() => setSyncModalOpen(false)}
          onLaunchReviewCandidates={() => void launchReviewCandidates()}
          onRetryReviewPreview={() => void retryReviewCandidatePreview()}
        />
      ) : null}

      <section className="page-heading reference-heading">
        <div>
          <p className="text-[13px] font-bold text-[var(--primary-dark)]">Tools</p>
          <h1>연동과 에이전트 도구</h1>
          <p>
            Slack, 메일, 문서, 캘린더 데이터를 공통 ingestion contract로 받아 Review Queue와 RAG 흐름에 연결합니다.
          </p>
        </div>
        <div className="panel inline-flex h-fit w-fit items-center gap-2 px-4 py-3 text-[13px] font-bold text-[var(--ink-subtle)]">
          <PlugZap className="h-4 w-4 text-[var(--workspace-accent)]" aria-hidden="true" />
          Connector contract ready
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_360px] items-start">
        <div className="grid gap-4 sm:grid-cols-2 items-start">
          {visibleManifests.map((manifest) => {
            const visual = integrationVisuals[manifest.type as keyof typeof integrationVisuals] ?? fallbackVisual;
            const Logo = visual.logo;
            const pending = pendingType === manifest.type;
            const connection = connections.find((item) => item.connector_type === manifest.type);
            const credentialAvailable = connection?.credential_status === "available";
            const oauthInstall = manifest.type === "slack" ? slackOAuth : googleOAuthByType[manifest.type];
            const canStartOAuth = Boolean(oauthInstall?.configured && (!connection || !credentialAvailable));
            const showOAuthStatus = manifest.auth_type === "oauth";
            const showSyncAction = !showOAuthStatus || Boolean(connection);
            const showDisconnectAction = Boolean(connection);
            const showDocumentsAction = manifest.type === "drive";
            const showMockModeNote = manifest.mode === "mock";
            const showCardFooter = showMockModeNote || showSyncAction || showDisconnectAction || showDocumentsAction;
            const oauthTheme =
              manifest.type === "slack"
                ? {
                    border: "border-[var(--line-soft)]",
                    bg: "bg-[#fbf8fd]",
                    icon: "text-[var(--workspace-accent)]",
                    text: "text-[var(--ink-strong)]",
                    pill: "text-[var(--ink-strong)]",
                    button: "border-[var(--line-soft)] text-[var(--ink-strong)] hover:bg-[#fbf8fd]",
                  }
                : {
                    border: "border-[var(--line-soft)]",
                    bg: "bg-blue-50/70",
                    icon: "text-[var(--workspace-accent)]",
                    text: "text-[var(--ink-strong)]",
                    pill: "text-[var(--ink-strong)]",
                    button: "border-[var(--line-soft)] text-[var(--ink-strong)] hover:bg-blue-50",
                  };
            return (
              <article
                key={manifest.type}
                className="integration-glass-card min-h-[24.5rem] rounded-lg border border-[var(--line-soft)] bg-[var(--glass-elevated)] p-4 shadow-sm transition hover:-translate-y-0.5 hover:shadow-md flex flex-col"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex min-w-0 items-start gap-3">
                    <span
                      className="grid h-11 w-11 shrink-0 place-items-center rounded-lg border border-[var(--line-soft)] bg-white shadow-sm"
                      data-testid={`${manifest.type}-connector-logo`}
                    >
                      <Logo />
                    </span>
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <h3 className="text-base font-semibold">{manifest.display_name}</h3>
                      </div>
                      <p className="mt-1 text-xs font-semibold text-[var(--workspace-rail-active)]">
                        {manifest.mode} · {manifest.sync_strategy}
                      </p>
                    </div>
                  </div>
                  <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-500" aria-hidden="true" />
                </div>

                <p className="mt-4 min-h-12 text-sm leading-6 text-[var(--ink-muted)]">{visual.description}</p>

                <div className="mt-3 grid gap-2 text-xs text-[var(--ink-muted)]">
                  <div className="flex items-center gap-2">
                    <KeyRound className="h-3.5 w-3.5" aria-hidden="true" />
                    {manifest.auth_type.toUpperCase()} · {formatScopes(manifest.required_scopes)}
                  </div>
                  <div className="flex items-start gap-2">
                    <ShieldCheck className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                    <span>{manifest.cost_policy}</span>
                  </div>
                </div>

                {showOAuthStatus ? (
                  <div
                    data-testid={`${manifest.type}-oauth-status`}
                    className={`mt-4 rounded-lg border ${oauthTheme.border} ${oauthTheme.bg} p-3 text-sm`}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex min-w-0 items-center gap-2">
                        <LockKeyhole className={`h-4 w-4 shrink-0 ${oauthTheme.icon}`} aria-hidden="true" />
                        <span
                          data-testid={`${manifest.type}-oauth-workspace-name`}
                          className={`truncate font-semibold ${oauthTheme.text}`}
                        >
                          {connection
                            ? connection.workspace_name
                            : oauthInstall?.configured
                              ? `${manifest.display_name} 연결 필요`
                              : "OAuth 설정 필요"}
                        </span>
                      </div>
                      <div className="flex shrink-0 items-center gap-2">
                        {connection ? (
                          <span
                            data-testid={`${manifest.type}-oauth-state-pill`}
                            className={`rounded-full bg-[var(--glass-elevated)] px-2 py-0.5 text-xs font-semibold ${oauthTheme.pill}`}
                          >
                            {credentialAvailable ? connection.status : "token missing"}
                          </span>
                        ) : null}
                        {canStartOAuth ? (
                          <button
                            type="button"
                            onClick={() => startOAuth(manifest.display_name, oauthInstall)}
                            className={`inline-flex h-8 items-center justify-center gap-1.5 rounded-lg border bg-[var(--glass-elevated)] px-2.5 text-xs font-semibold shadow-sm ${oauthTheme.button}`}
                          >
                            <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
                            {connection ? `${manifest.display_name} 재연결` : `${manifest.display_name} 연결`}
                          </button>
                        ) : null}
                      </div>
                    </div>
                    <p className="mt-2 text-xs leading-5 text-[var(--ink-muted)]">
                      {connection
                        ? credentialAvailable
                          ? `권한 ${connection.scopes.length.toLocaleString()}개 확인됨 · ${connection.masked_bot_token}`
                          : `권한 ${connection.scopes.length.toLocaleString()}개 확인됨 · backend 재시작으로 개발 vault 토큰이 비어 있어 재연결이 필요합니다.`
                        : oauthInstall?.configured
                          ? `${manifest.display_name} 설치 URL이 준비되었습니다. 설치 후에도 동기화는 변경분만 가져옵니다.`
                          : "환경 변수 설정 전까지는 mock 데이터로 안전하게 시연합니다."}
                    </p>
                  </div>
                ) : null}

                {showCardFooter ? (
                  <div className="mt-3 flex flex-col gap-2 border-t border-[var(--line-soft)] pt-3">
                    {showMockModeNote ? <span className="text-xs text-[var(--ink-muted)]">현재 mock 데이터 사용</span> : null}
                    <div data-testid={`${manifest.type}-card-actions`} className="flex min-h-9 flex-wrap content-end items-center justify-end gap-1.5">
                      {showSyncAction ? (
                        <button
                          type="button"
                          onClick={() => void startSync(manifest.type)}
                          disabled={Boolean(pendingType)}
                          className="liquid-primary inline-flex h-9 min-w-[5.25rem] items-center justify-center gap-1.5 rounded-[20px] px-2.5 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-55"
                        >
                          <RefreshCw className="h-4 w-4" aria-hidden="true" />
                          {pending ? "동기화 중" : "동기화"}
                        </button>
                      ) : null}
                      {showDisconnectAction ? (
                      <button
                        type="button"
                        onClick={() => void disconnect(manifest.type)}
                        className="liquid-control inline-flex h-9 min-w-[4.25rem] items-center justify-center gap-1.5 rounded-[20px] px-2.5 text-sm font-semibold text-red-600 hover:bg-red-50 disabled:cursor-not-allowed disabled:opacity-55"
                      >
                        <Trash2 className="h-4 w-4" aria-hidden="true" />
                        해제
                      </button>
                    ) : null}
                    {showDocumentsAction ? (
                      <a
                        href="/documents"
                        className="liquid-control inline-flex h-9 min-w-[5.25rem] items-center justify-center gap-1.5 rounded-[20px] px-2.5 text-sm font-semibold text-[var(--ink-strong)]"
                      >
                        <FileText className="h-4 w-4" aria-hidden="true" />
                        문서 현황
                      </a>
                    ) : null}
                    </div>
                  </div>
                ) : null}
              </article>
            );
          })}
        </div>

        <aside className="integration-glass-card rounded-lg border border-[var(--line-soft)] bg-[var(--glass-elevated)] shadow-panel">
          <div className="border-b border-[var(--line-soft)] px-4 py-4">
            <div className="flex items-center gap-2">
              <Radio className="h-4 w-4 text-[var(--workspace-rail-active)]" aria-hidden="true" />
              <h3 className="text-sm font-semibold">소스별 연동 현황</h3>
            </div>
            <p className="mt-1 text-xs text-[var(--ink-muted)]">수집된 근거의 소스 분포</p>
          </div>

          <div className="space-y-3 p-4">
            {syncProgress?.backgrounded ? (
              <BackgroundSyncProgressCard progress={syncProgress} onOpen={() => setSyncModalOpen(true)} />
            ) : null}
            <SourceOperationsPanel summary={dashboardSummary} />
          </div>
        </aside>
      </section>
    </div>
  );
}

function ResultMetric({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="glass-row rounded-lg p-3">
      <p className="text-xs text-[var(--ink-muted)]">{label}</p>
      <p className="mt-1 font-semibold">{typeof value === "number" ? value.toLocaleString() : value}</p>
    </div>
  );
}

function ProgressBar({ progressPct }: { progressPct: number }) {
  const clamped = Math.max(0, Math.min(100, Math.round(progressPct)));
  return (
    <div className="mt-3" data-testid="sync-progress-percent">
      <div className="flex items-center justify-between text-xs font-bold">
        <span>진행률</span>
        <span>{clamped}%</span>
      </div>
      <div className="mt-1 h-2 overflow-hidden rounded-full bg-white/70">
        <div className="h-full rounded-full bg-[var(--workspace-accent)]" style={{ width: `${clamped}%` }} />
      </div>
    </div>
  );
}

function SyncProgressModal({
  progress,
  onBackground,
  onClose,
  onLaunchReviewCandidates,
  onRetryReviewPreview,
}: {
  progress: SyncProgressState;
  onBackground: () => void;
  onClose: () => void;
  onLaunchReviewCandidates: () => void;
  onRetryReviewPreview: () => void;
}) {
  const isRunning = progress.status === "running";
  const isComplete = progress.status === "complete";
  const isError = progress.status === "error";
  const result = progress.result;
  const createdReviewItems = result?.created_review_items ?? 0;
  const pendingReviewCount = result?.pending_review_count ?? 0;
  const usesV2TruthfulCopy = progress.completionCopyMode !== "legacy";
  const hasLaunchPanel = Boolean(progress.reviewWorkflow);
  const canLaunchFromCompletion =
    progress.completionCopyMode === "v2_actionable" && hasLaunchPanel;
  const runningStages = usesV2TruthfulCopy ? V2_SYNC_RUNNING_STAGES : SYNC_RUNNING_STAGES;
  const changedSourceCount = result?.changed_source_refs?.length ?? 0;

  return (
    <div
      data-testid="sync-progress-backdrop"
      role="dialog"
      aria-modal="true"
      aria-labelledby="sync-progress-title"
      className="fixed inset-0 z-50 grid place-items-center bg-[#120b18]/72 px-4 backdrop-blur-sm"
    >
      <div
        data-testid="sync-progress-modal"
        className="max-h-[calc(100vh-2rem)] w-full max-w-md overflow-y-auto rounded-lg border border-[var(--line-soft)] bg-[var(--glass-elevated)] p-5 shadow-2xl"
      >
        <div className="flex items-start justify-between gap-4">
          <div className="flex min-w-0 items-start gap-3">
            <span className="grid h-10 w-10 shrink-0 place-items-center rounded-lg bg-[var(--glass-strong)] text-[var(--workspace-rail-active)]">
              {isComplete ? (
                <CheckCircle2 className="h-5 w-5" aria-hidden="true" />
              ) : (
                <RefreshCw className={`h-5 w-5 ${isRunning ? "animate-spin" : ""}`} aria-hidden="true" />
              )}
            </span>
            <div className="min-w-0">
              <h2 id="sync-progress-title" className="text-base font-semibold text-[var(--ink-strong)]">
                {isComplete
                  ? "동기화 완료"
                  : isError
                    ? `${progress.displayName} 동기화 실패`
                    : `${progress.displayName} 동기화 중`}
              </h2>
              <p data-testid="sync-modal-step" className="mt-1 text-sm leading-6 text-[var(--ink-muted)]">
                {isComplete
                  ? usesV2TruthfulCopy
                    ? canLaunchFromCompletion
                      ? `${progress.displayName} 변경 근거를 준비했습니다. 검토 후보는 아래에서 명시적으로 만듭니다.`
                      : (progress.completionNotice ?? `${progress.displayName} 변경 근거와 검토 후보 가능 여부를 확인하고 있습니다.`)
                    : `${progress.displayName} 데이터를 검토 큐에 반영했습니다.`
                  : isError
                    ? (progress.errorMessage ?? "동기화 중 오류가 발생했습니다.")
                    : progress.backgrounded
                      ? (progress.errorMessage ?? BACKGROUND_SYNC_CONTINUES_MESSAGE)
                      : runningStages[progress.stageIndex]}
              </p>
            </div>
          </div>

          {!isRunning ? (
            <button
              type="button"
              onClick={onClose}
              className="liquid-control grid h-8 w-8 shrink-0 place-items-center rounded-lg"
              aria-label="모달 닫기"
            >
              <X className="h-4 w-4" aria-hidden="true" />
            </button>
          ) : null}
        </div>

        <div className="mt-4 grid gap-2 text-sm">
          <SyncStep label="원본 수집" active={isRunning && progress.stageIndex === 0} done={progress.stageIndex > 0 || isComplete} />
          <SyncStep
            label={usesV2TruthfulCopy ? "변경 근거 준비" : "AI 분석"}
            active={isRunning && progress.stageIndex === 1}
            done={progress.stageIndex > 1 || isComplete}
          />
          <SyncStep
            label={usesV2TruthfulCopy ? "미리보기 준비" : "검토 항목 저장"}
            active={isRunning && progress.stageIndex === 2}
            done={isComplete}
          />
        </div>

        <ProgressBar progressPct={progress.progressPct} />
        {progress.lastMessage ? (
          <p className="mt-2 break-all text-xs leading-5 text-[var(--ink-muted)]">{progress.lastMessage}</p>
        ) : null}

        {result ? (
          usesV2TruthfulCopy ? (
            <>
              <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                <ResultMetric label="수집된 원본" value={`${result.fetched_events.toLocaleString()}개`} />
                <ResultMetric label="변경 근거" value={`${changedSourceCount.toLocaleString()}개`} />
              </div>
              <p className="mt-3 text-sm font-medium text-[var(--ink-strong)]">
                {canLaunchFromCompletion
                  ? `변경 근거 ${changedSourceCount.toLocaleString()}개를 준비했습니다. 검토 후보는 아래에서 만듭니다.`
                  : (progress.completionNotice ?? `변경 근거 ${changedSourceCount.toLocaleString()}개를 확인하고 있습니다.`)}
              </p>
            </>
          ) : (
            <>
            <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
              <ResultMetric label="새 검토 항목" value={`${createdReviewItems.toLocaleString()}개`} />
              <ResultMetric label="검토 대기" value={`${pendingReviewCount.toLocaleString()}개`} />
            </div>
            <p className="mt-3 text-sm font-medium text-[var(--ink-strong)]">
              새 검토 항목 {createdReviewItems.toLocaleString()}개, 검토 대기{" "}
              {pendingReviewCount.toLocaleString()}개입니다.
            </p>
            </>
          )
        ) : null}

        {progress.reviewWorkflow ? (
          <ReviewCandidateLaunchPanel
            diagnostic={progress.reviewWorkflow.diagnostic}
            dryRun={progress.reviewWorkflow.dryRun}
            launchState={progress.reviewWorkflow.launchState}
            terminalStatus={progress.reviewWorkflow.terminalStatus}
            errorCode={progress.reviewWorkflow.errorCode}
            errorMessage={progress.reviewWorkflow.errorMessage}
            onLaunch={onLaunchReviewCandidates}
            onRetryPreview={onRetryReviewPreview}
          />
        ) : null}

        <div className="mt-5 grid gap-2 sm:flex sm:flex-wrap sm:justify-end">
          {isRunning ? (
            <button
              type="button"
              onClick={onBackground}
              className="liquid-control inline-flex h-9 w-full items-center justify-center rounded-lg px-3 text-sm font-semibold sm:w-auto"
            >
              백그라운드에서 계속 진행
            </button>
          ) : null}
          {isComplete && progress.completionCopyMode === "legacy" && !progress.reviewWorkflow ? (
            <a
              href="/review"
              className="liquid-primary inline-flex h-9 w-full items-center justify-center gap-2 rounded-lg px-3 text-sm font-semibold sm:w-auto"
            >
              검토사항으로 이동
              <ArrowRight className="h-4 w-4" aria-hidden="true" />
            </a>
          ) : null}
          {!isRunning ? (
            <button
              type="button"
              onClick={onClose}
              className="liquid-control inline-flex h-9 w-full items-center justify-center rounded-lg px-3 text-sm font-semibold sm:w-auto"
            >
              닫기
            </button>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function SyncStep({ label, active, done }: { label: string; active: boolean; done: boolean }) {
  return (
    <div className="flex items-center gap-2 rounded-lg bg-[var(--glass-strong)] px-3 py-2">
      <span
        className={`grid h-5 w-5 shrink-0 place-items-center rounded-full text-[11px] font-bold ${
          done
            ? "bg-emerald-100 text-emerald-700"
            : active
              ? "bg-[var(--workspace-accent)] text-[#13231f]"
              : "bg-[var(--glass-elevated)] text-[var(--ink-muted)]"
        }`}
      >
        {done ? "✓" : active ? "…" : ""}
      </span>
      <span className="font-medium text-[var(--ink-strong)]">{label}</span>
    </div>
  );
}

/**
 * 소스별 데이터 수집 현황을 보여주는 패널 컴포넌트
 */
function BackgroundSyncProgressCard({
  progress,
  onOpen,
}: {
  progress: SyncProgressState;
  onOpen: () => void;
}) {
  const isRunning = progress.status === "running";
  const isComplete = progress.status === "complete";
  const statusLabel = isComplete ? "완료" : progress.status === "error" ? "오류" : "진행 중";
  const message = isComplete
    ? `${progress.displayName} 동기화가 완료되었습니다.`
    : progress.errorMessage || progress.lastMessage || SYNC_RUNNING_STAGES[progress.stageIndex];

  return (
    <div
      data-testid="background-sync-progress"
      className="rounded-lg border border-[var(--line-soft)] bg-[var(--glass-elevated)] p-3"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs font-bold text-[var(--workspace-rail-active)]">동기화 진행률</p>
          <h4 className="mt-1 truncate text-sm font-semibold text-[var(--ink-strong)]">
            {progress.displayName} {statusLabel}
          </h4>
        </div>
        <RefreshCw
          className={`h-4 w-4 shrink-0 text-[var(--workspace-accent)] ${isRunning ? "animate-spin" : ""}`}
          aria-hidden="true"
        />
      </div>
      <ProgressBar progressPct={progress.progressPct} />
      <p className="mt-2 line-clamp-2 text-xs leading-5 text-[var(--ink-muted)]">{message}</p>
      <button
        type="button"
        onClick={onOpen}
        className="liquid-control mt-3 inline-flex h-8 w-full items-center justify-center gap-1.5 rounded-lg px-3 text-xs font-semibold"
      >
        진행률 보기
        <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" />
      </button>
    </div>
  );
}

function SourceOperationsPanel({ summary }: { summary?: DashboardResponse }) {
  const counts = summary?.source_counts ?? {
    slack: 0,
    gmail: 0,
    drive: 0,
    calendar: 0,
    other: 0,
  };
  const total =
    (counts.slack ?? 0) +
    (counts.gmail ?? 0) +
    (counts.drive ?? 0) +
    (counts.calendar ?? 0);

  return (
    <div
      data-testid="source-operations-panel"
      className="rounded-lg border border-[var(--line-soft)] bg-[var(--glass-elevated)] p-3"
    >
      <div className="flex items-center justify-between gap-3">
        <div>
          <h4 className="text-sm font-semibold text-[var(--ink-strong)]">소스별 연동 현황</h4>
          <p className="mt-1 text-xs text-[var(--ink-muted)]">수집된 근거 수</p>
        </div>
        <span className="rounded-full bg-[var(--glass-strong)] px-2 py-1 text-xs font-semibold text-[var(--ink-muted)]">
          총 {total.toLocaleString()}
        </span>
      </div>
      <div className="mt-3 space-y-2">
        <SourceOperationRow id="slack" label="Slack" value={counts.slack ?? 0} total={total} tone="purple" />
        <SourceOperationRow id="gmail" label="Gmail" value={counts.gmail ?? 0} total={total} tone="blue" />
        <SourceOperationRow id="drive" label="Google Drive" value={counts.drive ?? 0} total={total} tone="green" />
        <SourceOperationRow id="calendar" label="Google Calendar" value={counts.calendar ?? 0} total={total} tone="orange" />
      </div>
    </div>
  );
}

function SourceOperationRow({
  id,
  label,
  value,
  total,
  tone,
}: {
  id: string;
  label: string;
  value: number;
  total: number;
  tone: string;
}) {
  const percent = total > 0 ? Math.round((value / total) * 100) : 0;
  return (
    <div className="source-bar-row" data-testid={`source-operation-${id}`}>
      <div className="source-bar-row-header" data-testid={`source-operation-${id}-header`}>
        <span>{label}</span>
        <strong data-testid={`source-operation-${id}-count`}>{value.toLocaleString()}</strong>
      </div>
      <div className="source-bar-track" data-testid={`source-operation-${id}-bar`}>
        <i className={tone} style={{ width: `${Math.max(percent, value > 0 ? 4 : 0)}%` }} />
      </div>
    </div>
  );
}

function SlackLogo() {
  return (
    <svg className="h-7 w-7" viewBox="0 0 127 127" aria-hidden="true" data-logo-source="official-slack-mark">
      <path
        d="M27.2 80c0 7.3-5.9 13.2-13.2 13.2C6.7 93.2.8 87.3.8 80c0-7.3 5.9-13.2 13.2-13.2h13.2V80zm6.6 0c0-7.3 5.9-13.2 13.2-13.2 7.3 0 13.2 5.9 13.2 13.2v33c0 7.3-5.9 13.2-13.2 13.2-7.3 0-13.2-5.9-13.2-13.2V80z"
        fill="#E01E5A"
      />
      <path
        d="M47 27c-7.3 0-13.2-5.9-13.2-13.2C33.8 6.5 39.7.6 47 .6c7.3 0 13.2 5.9 13.2 13.2V27H47zm0 6.7c7.3 0 13.2 5.9 13.2 13.2 0 7.3-5.9 13.2-13.2 13.2H13.9C6.6 60.1.7 54.2.7 46.9c0-7.3 5.9-13.2 13.2-13.2H47z"
        fill="#36C5F0"
      />
      <path
        d="M99.9 46.9c0-7.3 5.9-13.2 13.2-13.2 7.3 0 13.2 5.9 13.2 13.2 0 7.3-5.9 13.2-13.2 13.2H99.9V46.9zm-6.6 0c0 7.3-5.9 13.2-13.2 13.2-7.3 0-13.2-5.9-13.2-13.2V13.8C66.9 6.5 72.8.6 80.1.6c7.3 0 13.2 5.9 13.2 13.2v33.1z"
        fill="#2EB67D"
      />
      <path
        d="M80.1 99.8c7.3 0 13.2 5.9 13.2 13.2 0 7.3-5.9 13.2-13.2 13.2-7.3 0-13.2-5.9-13.2-13.2V99.8h13.2zm0-6.6c-7.3 0-13.2-5.9-13.2-13.2 0-7.3 5.9-13.2 13.2-13.2h33.1c7.3 0 13.2 5.9 13.2 13.2 0 7.3-5.9 13.2-13.2 13.2H80.1z"
        fill="#ECB22E"
      />
    </svg>
  );
}

function GmailLogo() {
  return (
    <svg className="h-7 w-8" viewBox="52 42 88 66" aria-hidden="true" data-logo-source="official-gmail-2020">
      <path fill="#4285f4" d="M58 108h14V74L52 59v43c0 3.32 2.69 6 6 6" />
      <path fill="#34a853" d="M120 108h14c3.32 0 6-2.69 6-6V59l-20 15" />
      <path fill="#fbbc04" d="M120 48v26l20-15v-8c0-7.42-8.47-11.65-14.4-7.2" />
      <path fill="#ea4335" d="M72 74V48l24 18 24-18v26L96 92" />
      <path fill="#c5221f" d="M52 51v8l20 15V48l-5.6-4.2c-5.94-4.45-14.4-.22-14.4 7.2" />
    </svg>
  );
}

function GoogleDriveLogo() {
  return (
    <svg className="h-7 w-8" viewBox="0 0 87.3 78" aria-hidden="true" data-logo-source="official-google-drive-2020">
      <path
        d="m6.6 66.85 3.85 6.65c.8 1.4 1.95 2.5 3.3 3.3l13.75-23.8h-27.5c0 1.55.4 3.1 1.2 4.5z"
        fill="#0066da"
      />
      <path
        d="m43.65 25-13.75-23.8c-1.35.8-2.5 1.9-3.3 3.3l-25.4 44a9.06 9.06 0 0 0 -1.2 4.5h27.5z"
        fill="#00ac47"
      />
      <path
        d="m73.55 76.8c1.35-.8 2.5-1.9 3.3-3.3l1.6-2.75 7.65-13.25c.8-1.4 1.2-2.95 1.2-4.5h-27.502l5.852 11.5z"
        fill="#ea4335"
      />
      <path
        d="m43.65 25 13.75-23.8c-1.35-.8-2.9-1.2-4.5-1.2h-18.5c-1.6 0-3.15.45-4.5 1.2z"
        fill="#00832d"
      />
      <path
        d="m59.8 53h-32.3l-13.75 23.8c1.35.8 2.9 1.2 4.5 1.2h50.8c1.6 0 3.15-.45 4.5-1.2z"
        fill="#2684fc"
      />
      <path
        d="m73.4 26.5-12.7-22c-.8-1.4-1.95-2.5-3.3-3.3l-13.75 23.8 16.15 28h27.45c0-1.55-.4-3.1-1.2-4.5z"
        fill="#ffba00"
      />
    </svg>
  );
}

function GoogleCalendarLogo() {
  return (
    <svg className="h-7 w-7" viewBox="0 0 200 200" aria-hidden="true" data-logo-source="official-google-calendar-2020">
      <g transform="translate(3.75 3.75)">
        <path
          fill="#FFFFFF"
          d="M148.882,43.618l-47.368-5.263l-57.895,5.263L38.355,96.25l5.263,52.632l52.632,6.579l52.632-6.579l5.263-53.947L148.882,43.618z"
        />
        <path
          fill="#1A73E8"
          d="M65.211,125.276c-3.934-2.658-6.658-6.539-8.145-11.671l9.132-3.763c0.829,3.158,2.276,5.605,4.342,7.342c2.053,1.737,4.553,2.592,7.474,2.592c2.987,0,5.553-0.908,7.697-2.724s3.224-4.132,3.224-6.934c0-2.868-1.132-5.211-3.395-7.026s-5.105-2.724-8.5-2.724h-5.276v-9.039H76.5c2.921,0,5.382-0.789,7.382-2.368c2-1.579,3-3.737,3-6.487c0-2.447-0.895-4.395-2.684-5.855s-4.053-2.197-6.803-2.197c-2.684,0-4.816,0.711-6.395,2.145s-2.724,3.197-3.447,5.276l-9.039-3.763c1.197-3.395,3.395-6.395,6.618-8.987c3.224-2.592,7.342-3.895,12.342-3.895c3.697,0,7.026,0.711,9.974,2.145c2.947,1.434,5.263,3.421,6.934,5.947c1.671,2.539,2.5,5.382,2.5,8.539c0,3.224-0.776,5.947-2.329,8.184c-1.553,2.237-3.461,3.947-5.724,5.145v0.539c2.987,1.25,5.421,3.158,7.342,5.724c1.908,2.566,2.868,5.632,2.868,9.211s-0.908,6.776-2.724,9.579c-1.816,2.803-4.329,5.013-7.513,6.618c-3.197,1.605-6.789,2.421-10.776,2.421C73.408,129.263,69.145,127.934,65.211,125.276z"
        />
        <path fill="#1A73E8" d="M121.25,79.961l-9.974,7.25l-5.013-7.605l17.987-12.974h6.895v61.197h-9.895L121.25,79.961z" />
        <path fill="#EA4335" d="M148.882,196.25l47.368-47.368l-23.684-10.526l-23.684,10.526l-10.526,23.684L148.882,196.25z" />
        <path fill="#34A853" d="M33.092,172.566l10.526,23.684h105.263v-47.368H43.618L33.092,172.566z" />
        <path
          fill="#4285F4"
          d="M12.039-3.75C3.316-3.75-3.75,3.316-3.75,12.039v136.842l23.684,10.526l23.684-10.526V43.618h105.263l10.526-23.684L148.882-3.75H12.039z"
        />
        <path fill="#188038" d="M-3.75,148.882v31.579c0,8.724,7.066,15.789,15.789,15.789h31.579v-47.368H-3.75z" />
        <path fill="#FBBC04" d="M148.882,43.618v105.263h47.368V43.618l-23.684-10.526L148.882,43.618z" />
        <path fill="#1967D2" d="M196.25,43.618V12.039c0-8.724-7.066-15.789-15.789-15.789h-31.579v47.368H196.25z" />
      </g>
    </svg>
  );
}

function FallbackConnectorLogo() {
  return <PlugZap className="h-5 w-5 text-[var(--workspace-accent)]" aria-hidden="true" />;
}

function formatScopes(scopes: string[]) {
  if (scopes.length === 0) {
    return "scope 준비 중";
  }
  if (scopes.length <= 2) {
    return scopes.join(", ");
  }
  return `${scopes[0]}, ${scopes[1]} 외 ${scopes.length - 2}개`;
}

function connectorDisplayName(type: string, manifests: IntegrationManifest[]) {
  return manifests.find((manifest) => manifest.type === type)?.display_name ?? formatConnectorName(type);
}

function formatConnectorName(connectorType: string) {
  if (connectorType === "slack") {
    return "Slack";
  }
  if (connectorType === "gmail") {
    return "Gmail";
  }
  if (connectorType === "drive") {
    return "Google Drive";
  }
  if (connectorType === "calendar") {
    return "Google Calendar";
  }
  return connectorType;
}
