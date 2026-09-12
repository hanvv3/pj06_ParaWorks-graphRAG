import type { AssistantMessage } from "../api/types";

export type DeliveryStatus = "sending" | "persisted" | "unknown";

export type DeliveryOwner = {
  requestToken: number;
  conversationId: number;
  optimisticMessageId: number;
};

const STATUS_KEY = "ui_delivery_status";
const TOKEN_KEY = "ui_request_token";

export function markDelivery<T extends AssistantMessage>(
  message: T,
  status: DeliveryStatus,
  requestToken: number,
): T {
  return {
    ...message,
    metadata: {
      ...message.metadata,
      [STATUS_KEY]: status,
      [TOKEN_KEY]: requestToken,
    },
  };
}

export function getDeliveryStatus(message: AssistantMessage): DeliveryStatus | null {
  const status = message.metadata?.[STATUS_KEY];
  return status === "sending" || status === "persisted" || status === "unknown"
    ? status
    : null;
}

export function isCurrentDelivery(
  owner: DeliveryOwner,
  currentRequestToken: number,
  activeConversationId: number | undefined,
): boolean {
  return owner.requestToken === currentRequestToken
    && owner.conversationId === activeConversationId;
}

export function removeOwnedOptimistic(
  messages: AssistantMessage[],
  owner: DeliveryOwner,
): AssistantMessage[] {
  return messages.filter((message) => !(
    message.id === owner.optimisticMessageId
    && message.conversation_id === owner.conversationId
    && message.metadata?.[TOKEN_KEY] === owner.requestToken
  ));
}

export function markOwnedOptimisticUnknown(
  messages: AssistantMessage[],
  owner: DeliveryOwner,
): AssistantMessage[] {
  return messages.map((message) => (
    message.id === owner.optimisticMessageId
      && message.conversation_id === owner.conversationId
      && message.metadata?.[TOKEN_KEY] === owner.requestToken
      ? markDelivery(message, "unknown", owner.requestToken)
      : message
  ));
}
