const RAG_AGENT_NAME = "rag_orchestrator_agent";
const LEGACY_RAG_PROMPT_VERSION = "rag-answer:v1";
const ABSOLUTE_HTTP_URL = /^https?:\/\//i;
const UNSAFE_URL_CHARACTERS = /[\s\u0000-\u001f\u007f-\u009f]/u;

export type RagPresentationMetadata = {
  agent_name?: unknown;
  prompt_version?: unknown;
};

export function isPlainTextRagMessage(metadata: RagPresentationMetadata): boolean {
  return metadata.agent_name === RAG_AGENT_NAME
    && metadata.prompt_version !== LEGACY_RAG_PROMPT_VERSION;
}

export function isProvenLegacyRagMessage(metadata: RagPresentationMetadata): boolean {
  return metadata.agent_name === RAG_AGENT_NAME
    && metadata.prompt_version === LEGACY_RAG_PROMPT_VERSION;
}

export function isSafeCitationUrl(value: unknown): value is string {
  if (
    typeof value !== "string"
    || value.length === 0
    || !ABSOLUTE_HTTP_URL.test(value)
    || UNSAFE_URL_CHARACTERS.test(value)
  ) {
    return false;
  }

  try {
    const parsed = new URL(value);
    return (parsed.protocol === "http:" || parsed.protocol === "https:")
      && parsed.username === ""
      && parsed.password === "";
  } catch {
    return false;
  }
}

export function publicPermissionNotice(metadata: {
  permission_notice: string | null;
  hidden_match_count: number;
}): string | null {
  if (metadata.permission_notice === "evidence_unavailable") {
    return "이 답변의 근거를 더 이상 확인할 수 없습니다. 다시 생성해 주세요.";
  }
  if (metadata.hidden_match_count > 0 || metadata.permission_notice !== null) {
    return "일부 근거는 권한에 따라 숨겨졌습니다.";
  }
  return null;
}
