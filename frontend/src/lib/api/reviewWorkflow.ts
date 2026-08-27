import { apiGet, apiPost } from "./client";
import type {
  ReviewWorkflowDiagnostic,
  ReviewWorkflowDryRun,
  ReviewWorkflowErrorCode,
  ReviewWorkflowRunRequest,
  ReviewWorkflowSourceRef,
  ReviewWorkflowStatus,
} from "./types";

const REVIEW_WORKFLOW_PATH = "/api/v1/orchestration/v2/company-memory";
const GENERIC_REVIEW_WORKFLOW_ERROR = "요청을 처리할 수 없습니다. 잠시 후 다시 시도해 주세요.";
const MAX_SERIALIZED_ERROR_LENGTH = 512;

const REVIEW_WORKFLOW_ERROR_CODES = new Set<ReviewWorkflowErrorCode>([
  "invalid_input",
  "not_found",
  "idempotency_key_reused",
  "evidence_changed",
  "permission_denied",
  "checkpoint_unavailable",
  "checkpoint_failed",
  "review_unresolved",
  "runtime_version_unavailable",
  "model_unavailable",
  "budget_exceeded",
  "concurrent_resume",
  "invalid_state_transition",
]);

export class ReviewWorkflowClientError extends Error {
  readonly code: ReviewWorkflowErrorCode | null;

  constructor(code: ReviewWorkflowErrorCode | null) {
    super(GENERIC_REVIEW_WORKFLOW_ERROR);
    this.name = "ReviewWorkflowClientError";
    this.code = code;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isKnownErrorCode(value: unknown): value is ReviewWorkflowErrorCode {
  return typeof value === "string" && REVIEW_WORKFLOW_ERROR_CODES.has(value as ReviewWorkflowErrorCode);
}

function readSerializedCode(value: unknown): ReviewWorkflowErrorCode | null {
  if (!isRecord(value)) return null;

  const keys = Object.keys(value);
  if (keys.length === 1 && keys[0] === "code" && isKnownErrorCode(value.code)) {
    return value.code;
  }
  return null;
}

/** Converts bounded V2 API failures into a stable, display-safe UI error. */
export function readReviewWorkflowError(error: unknown): ReviewWorkflowClientError {
  if (!(error instanceof Error) || error.message.length > MAX_SERIALIZED_ERROR_LENGTH) {
    return new ReviewWorkflowClientError(null);
  }

  try {
    return new ReviewWorkflowClientError(readSerializedCode(JSON.parse(error.message)));
  } catch {
    return new ReviewWorkflowClientError(null);
  }
}

function runRequestBody(request: ReviewWorkflowRunRequest): ReviewWorkflowRunRequest {
  const source_refs: ReviewWorkflowSourceRef[] = request.source_refs.map((source) => ({
    source_type: source.source_type,
    source_id: source.source_id,
    version_or_signature: source.version_or_signature,
  }));
  const body: ReviewWorkflowRunRequest = {
    source_refs,
    agent_names: [...request.agent_names],
  };
  if (request.client_request_id !== undefined) {
    body.client_request_id = request.client_request_id;
  }
  return body;
}

async function boundedRequest<T>(request: Promise<T>): Promise<T> {
  try {
    return await request;
  } catch (error) {
    throw readReviewWorkflowError(error);
  }
}

function runPath(threadId: string): string {
  return `${REVIEW_WORKFLOW_PATH}/runs/${encodeURIComponent(threadId)}`;
}

export function getReviewWorkflowDiagnostic(): Promise<ReviewWorkflowDiagnostic> {
  return boundedRequest(apiGet<ReviewWorkflowDiagnostic>(REVIEW_WORKFLOW_PATH));
}

export function dryRunReviewWorkflow(request: ReviewWorkflowRunRequest): Promise<ReviewWorkflowDryRun> {
  return boundedRequest(apiPost<ReviewWorkflowDryRun>(`${REVIEW_WORKFLOW_PATH}/dry-run`, runRequestBody(request)));
}

export function launchReviewWorkflow(request: ReviewWorkflowRunRequest): Promise<ReviewWorkflowStatus> {
  return boundedRequest(apiPost<ReviewWorkflowStatus>(`${REVIEW_WORKFLOW_PATH}/runs`, runRequestBody(request)));
}

export function getReviewWorkflowStatus(threadId: string): Promise<ReviewWorkflowStatus> {
  return boundedRequest(apiGet<ReviewWorkflowStatus>(runPath(threadId)));
}

export function resumeReviewWorkflow(threadId: string): Promise<ReviewWorkflowStatus> {
  return boundedRequest(apiPost<ReviewWorkflowStatus>(`${runPath(threadId)}/resume`));
}

export function cancelReviewWorkflow(threadId: string): Promise<ReviewWorkflowStatus> {
  return boundedRequest(apiPost<ReviewWorkflowStatus>(`${runPath(threadId)}/cancel`));
}
