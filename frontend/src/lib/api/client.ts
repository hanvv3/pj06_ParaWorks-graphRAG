export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
export const DEMO_USER_STORAGE_KEY = "paraworks-demo-user";
export const DEFAULT_DEMO_USER = "hanvv-employee";

export const LOCAL_DEMO_USERS = [
  {
    id: "demo-admin",
    email: "admin@paraworks.com",
    role: "admin",
    permission_levels: ["public", "internal", "restricted"],
    name: "ParaWorks Admin",
    title: "Workspace Administrator",
    department: "Platform",
  },
  {
    id: "google-hanvv-admin",
    email: "hanvv3@gmail.com",
    role: "admin",
    permission_levels: ["public", "internal", "restricted"],
    name: "Hanvv Admin",
    title: "Workspace Administrator",
    department: "Platform",
  },
  {
    id: "kjw4work",
    email: "kjw4work@gmail.com",
    role: "admin",
    permission_levels: ["public", "internal", "restricted"],
    name: "Kim Jongwoo",
    title: "COO",
    department: "platform",
  },
  {
    id: "yonghee199702",
    email: "yonghee199702@gmail.com",
    role: "admin",
    permission_levels: ["public", "internal", "restricted"],
    name: "Kim Yonghee",
    title: "CTO",
    department: "platform",
  },
  {
    id: "google-hanvv-employee",
    email: "hanvv3@koreacu.ac.kr",
    role: "employee",
    permission_levels: ["public", "internal"],
    name: "Hanvv Employee",
    title: "AI Agent Developer",
    department: "Engineering",
  },
  {
    id: "employee-mina",
    email: "mina@paraworks.com",
    role: "reviewer",
    permission_levels: ["public", "internal"],
    name: "Kim Mina",
    title: "Product Manager",
    department: "Product",
  },
];

function apiUrl(path: string) {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return typeof window === "undefined" ? `${API_BASE}${normalizedPath}` : normalizedPath;
}

export function getStoredDemoUserId() {
  if (typeof window === "undefined") {
    return DEFAULT_DEMO_USER;
  }

  return window.localStorage.getItem(DEMO_USER_STORAGE_KEY) || DEFAULT_DEMO_USER;
}

export function setStoredDemoUserId(userId: string) {
  window.localStorage.setItem(DEMO_USER_STORAGE_KEY, userId);
}

export function clearStoredDemoUserId() {
  window.localStorage.removeItem(DEMO_USER_STORAGE_KEY);
}

function demoUserHeader(demoUser?: string) {
  return demoUser ?? getStoredDemoUserId();
}

function getCookie(name: string): string | undefined {
  if (typeof document === "undefined") return undefined;
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop()?.split(";").shift();
  return undefined;
}

function csrfHeader(): Record<string, string> {
  const token = getCookie("paraworks_csrf");
  return token ? { "X-CSRF-Token": token } : {};
}

const SAFE_API_ERROR_MESSAGES = {
  input_safety_blocked: "민감한 정보로 보이는 내용이 포함되어 요청을 전송하지 않았습니다.",
  input_scanner_unavailable: "요청 내용을 확인하는 기능을 사용할 수 없습니다. 잠시 후 다시 시도해 주세요.",
  permission_denied: "이 요청을 처리할 권한이 없습니다.",
  budget_exceeded: "요청이 비용 한도를 초과해 답변을 생성하지 않았습니다.",
  runtime_version_unavailable: "현재 답변 기능을 사용할 수 없습니다. 잠시 후 다시 시도해 주세요.",
  retriever_not_configured: "검색 기능이 준비되지 않았습니다.",
  retriever_unavailable: "검색 기능을 사용할 수 없습니다. 잠시 후 다시 시도해 주세요.",
  model_unavailable: "답변 모델을 사용할 수 없습니다. 잠시 후 다시 시도해 주세요.",
  provider_safety_unavailable: "안전 확인 기능을 사용할 수 없습니다. 잠시 후 다시 시도해 주세요.",
  provider_response_identity_invalid: "답변을 안전하게 확인하지 못했습니다. 잠시 후 다시 시도해 주세요.",
  provider_usage_overrun: "요청이 사용 한도를 초과해 중단되었습니다.",
  provider_embedding_payload_invalid: "검색 요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.",
  model_provider_failed: "답변을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요.",
  structured_output_invalid: "답변 형식을 확인하지 못했습니다. 잠시 후 다시 시도해 주세요.",
  citation_validation_failed: "답변의 근거를 확인하지 못했습니다. 잠시 후 다시 시도해 주세요.",
  persistence_failed: "요청 처리 상태를 확인하지 못했습니다. 잠시 후 다시 시도해 주세요.",
  unexpected_internal_error: "요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.",
  client_upgrade_required: "새 버전이 필요합니다. 페이지를 새로고침해 주세요.",
} as const;

const REVIEW_WORKFLOW_API_ERROR_CODES = {
  invalid_input: true,
  not_found: true,
  idempotency_key_reused: true,
  evidence_changed: true,
  permission_denied: true,
  checkpoint_unavailable: true,
  checkpoint_failed: true,
  review_unresolved: true,
  runtime_version_unavailable: true,
  model_unavailable: true,
  budget_exceeded: true,
  cost_preview_changed: true,
  concurrent_resume: true,
  invalid_state_transition: true,
} as const;

// This consumer currently distinguishes only the durable remediation boundary.
// Other Auto-Review conflict codes intentionally remain generic UI failures.
const AUTO_REVIEW_API_ERROR_CODES = {
  remediation_required: true,
} as const;

function hasOwnKey<T extends object>(value: T, key: PropertyKey): key is keyof T {
  return Object.prototype.hasOwnProperty.call(value, key);
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string | null,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

const UNKNOWN_API_ERROR_MESSAGE = "요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.";
const MAX_API_ERROR_BODY_LENGTH = 2_048;

export function decodeApiErrorEnvelope(payload: unknown): string | null {
  try {
    if (typeof payload !== "object" || payload === null || Array.isArray(payload)) return null;
    const rootKeys = Reflect.ownKeys(payload);
    if (rootKeys.length !== 1 || rootKeys[0] !== "detail") return null;
    const detailDescriptor = Object.getOwnPropertyDescriptor(payload, "detail");
    if (detailDescriptor === undefined || !hasOwnKey(detailDescriptor, "value")) return null;
    const detail = detailDescriptor.value;
    if (typeof detail !== "object" || detail === null || Array.isArray(detail)) return null;
    const detailKeys = Reflect.ownKeys(detail);
    if (detailKeys.length !== 1 || detailKeys[0] !== "code") return null;
    const codeDescriptor = Object.getOwnPropertyDescriptor(detail, "code");
    if (codeDescriptor === undefined || !hasOwnKey(codeDescriptor, "value")) return null;
    const code = codeDescriptor.value;
    if (typeof code !== "string") return null;
    if (
      hasOwnKey(SAFE_API_ERROR_MESSAGES, code)
      || hasOwnKey(REVIEW_WORKFLOW_API_ERROR_CODES, code)
      || hasOwnKey(AUTO_REVIEW_API_ERROR_CODES, code)
    ) {
      return code;
    }
  } catch {
    // Reject proxy/accessor-shaped values without reading or retaining them.
  }
  return null;
}

function decodeApiErrorBody(rawBody: string): string | null {
  if (rawBody.length === 0 || rawBody.length > MAX_API_ERROR_BODY_LENGTH) return null;
  try {
    return decodeApiErrorEnvelope(JSON.parse(rawBody) as unknown);
  } catch {
    return null;
  }
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const rawBody = await response.text().catch(() => "");
    const code = decodeApiErrorBody(rawBody);
    throw new ApiError(
      response.status,
      code,
      code !== null && hasOwnKey(SAFE_API_ERROR_MESSAGES, code)
        ? SAFE_API_ERROR_MESSAGES[code]
        : UNKNOWN_API_ERROR_MESSAGE,
    );
  }

  return response.json() as Promise<T>;
}

export async function apiGet<T>(
  path: string,
  demoUser?: string,
  additionalHeaders?: HeadersInit,
): Promise<T> {
  const headers = new Headers({
    "X-Demo-User": demoUserHeader(demoUser),
  });
  new Headers(additionalHeaders).forEach((value, name) => headers.set(name, value));
  const response = await fetch(apiUrl(path), {
    headers,
    credentials: "include",
    cache: "no-store",
  });

  return parseResponse<T>(response);
}

export async function apiPost<T>(
  path: string,
  body?: unknown,
  demoUser?: string,
  additionalHeaders?: HeadersInit,
): Promise<T> {
  const headers = new Headers({
    "Content-Type": "application/json",
    "X-Demo-User": demoUserHeader(demoUser),
    ...csrfHeader(),
  });
  new Headers(additionalHeaders).forEach((value, name) => headers.set(name, value));
  const response = await fetch(apiUrl(path), {
    method: "POST",
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    credentials: "include",
    cache: "no-store",
  });

  return parseResponse<T>(response);
}

export async function apiPatch<T>(
  path: string,
  body: unknown,
  demoUser?: string,
): Promise<T> {
  const response = await fetch(apiUrl(path), {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      "X-Demo-User": demoUserHeader(demoUser),
      ...csrfHeader(),
    },
    body: JSON.stringify(body),
    credentials: "include",
    cache: "no-store",
  });

  return parseResponse<T>(response);
}
export async function apiDelete<T>(
  path: string,
  demoUser?: string,
): Promise<T> {
  const response = await fetch(apiUrl(path), {
    method: "DELETE",
    headers: {
      "X-Demo-User": demoUserHeader(demoUser),
      ...csrfHeader(),
    },
    credentials: "include",
    cache: "no-store",
  });

  return parseResponse<T>(response);
}
