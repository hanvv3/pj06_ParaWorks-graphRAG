import { apiGet, apiPost } from "./client";
import type {
  AssistantConversationCreatedResponse,
  AssistantConversationsResponse,
  AssistantEmailSendResponse,
  AssistantMessagesResponse,
  AssistantTurnResponse,
} from "./types";

export const RAG_RENDER_CAPABILITY = "rag-v2-plain-text-citations:v1" as const;

const RAG_RENDER_CAPABILITY_HEADER = "X-ParaWorks-Rag-Render-Capability";
const RAG_RENDER_HEADERS = {
  [RAG_RENDER_CAPABILITY_HEADER]: RAG_RENDER_CAPABILITY,
} as const;

export function listAssistantConversations(demoUser?: string) {
  return apiGet<AssistantConversationsResponse>(
    "/api/v1/assistant/conversations",
    demoUser,
    RAG_RENDER_HEADERS,
  );
}

export function createAssistantConversation(title: string, demoUser?: string) {
  return apiPost<AssistantConversationCreatedResponse>(
    "/api/v1/assistant/conversations",
    { title },
    demoUser,
    RAG_RENDER_HEADERS,
  );
}

export function getAssistantMessages(conversationId: number, demoUser?: string) {
  return apiGet<AssistantMessagesResponse>(
    `/api/v1/assistant/conversations/${conversationId}/messages`,
    demoUser,
    RAG_RENDER_HEADERS,
  );
}

export function createAssistantMessage(
  conversationId: number,
  content: string,
  demoUser?: string,
) {
  return apiPost<AssistantTurnResponse>(
    `/api/v1/assistant/conversations/${conversationId}/messages`,
    { content },
    demoUser,
    RAG_RENDER_HEADERS,
  );
}

export function sendAssistantEmailDraft(messageId: number, demoUser?: string) {
  return apiPost<AssistantEmailSendResponse>(
    `/api/v1/assistant/messages/${messageId}/email/send`,
    undefined,
    demoUser,
    RAG_RENDER_HEADERS,
  );
}
