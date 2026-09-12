"use client";

import {
  Bot,
  ChevronDown,
  ChevronRight,
  Copy,
  FileText,
  Link2,
  MessageSquareText,
  PanelLeftClose,
  Plus,
  Send,
  ShieldAlert,
  Sparkles,
} from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { FormEvent, Suspense, useCallback, useEffect, useRef, useState } from "react";
import {
  createAssistantConversation,
  createAssistantMessage,
  getAssistantMessages,
  listAssistantConversations,
  sendAssistantEmailDraft,
} from "@/lib/api/assistant";
import { ApiError } from "@/lib/api/client";
import { ephemeralSearchHandoff } from "@/lib/assistant/searchHandoff";
import {
  type DeliveryOwner,
  getDeliveryStatus,
  isCurrentDelivery,
  markDelivery,
  markOwnedOptimisticUnknown,
  removeOwnedOptimistic,
} from "@/lib/assistant/reconciliation";
import {
  isPlainTextRagMessage,
  isSafeCitationUrl,
  publicPermissionNotice,
} from "@/lib/rag/presentation";
import type {
  AssistantConversation,
  AssistantMessage,
  RagCitation,
} from "@/lib/api/types";

const DEFAULT_CONVERSATION_TITLE = "새 대화";
const SUGGESTED_QUESTIONS = [
  "기획팀 회의 일정을 정리해줘",
  "최근 결정된 사항만 요약해줘",
  "내가 확인해야 할 할 일을 알려줘",
  "관련 근거가 있는 문서만 찾아줘",
];
const ASSISTANT_TYPING_INTERVAL_MS = 18;
const ASSISTANT_TYPING_CHUNK_SIZE = 2;

export default function SearchPage() {
  return (
    <Suspense fallback={<SearchPageFallback />}>
      <SearchPageContent />
    </Suspense>
  );
}

function SearchPageContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const [conversations, setConversations] = useState<AssistantConversation[]>([]);
  const [activeConversation, setActiveConversation] = useState<AssistantConversation>();
  const [messages, setMessages] = useState<AssistantMessage[]>([]);
  const [openEvidenceMessageIds, setOpenEvidenceMessageIds] = useState<Set<number>>(new Set());
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [booting, setBooting] = useState(true);
  const [error, setError] = useState<string>();
  const [unknownDelivery, setUnknownDelivery] = useState<{
    owner: DeliveryOwner;
    failureStatus: number;
    failureCopy: string;
  }>();
  const [clientUpgradeRequired, setClientUpgradeRequired] = useState(false);
  const [copiedMessageId, setCopiedMessageId] = useState<number>();
  const [sendingEmailMessageId, setSendingEmailMessageId] = useState<number>();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true);
  const [hydrated, setHydrated] = useState(false);
  const loadingRef = useRef(false);
  const creatingConversationRef = useRef(false);
  const loadMessagesRequestRef = useRef(0);
  const activeConversationIdRef = useRef<number | undefined>(undefined);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const nextOptimisticMessageIdRef = useRef(-1);
  const deliveryRequestTokenRef = useRef(0);
  const handoffConsumedRef = useRef(false);
  const typingTimerRef = useRef<number | undefined>(undefined);

  const upsertConversationByUpdatedAt = useCallback((conversation: AssistantConversation) => {
    setConversations((current) => sortConversationsByUpdatedAt([
      conversation,
      ...current.filter((item) => item.id !== conversation.id),
    ]));
  }, []);

  const createConversation = useCallback(async (title?: string) => {
    deliveryRequestTokenRef.current += 1;
    const requestId = ++loadMessagesRequestRef.current;
    const response = await createAssistantConversation(
      title?.trim() || DEFAULT_CONVERSATION_TITLE,
    );
    if (requestId === loadMessagesRequestRef.current) {
      activeConversationIdRef.current = response.conversation.id;
      setActiveConversation(response.conversation);
      setMessages([]);
      setUnknownDelivery(undefined);
      setClientUpgradeRequired(false);
      setOpenEvidenceMessageIds(new Set());
      upsertConversationByUpdatedAt(response.conversation);
    }
    return response.conversation;
  }, [upsertConversationByUpdatedAt]);

  const revealAssistantMessage = useCallback((message: AssistantMessage) => {
    if (typingTimerRef.current !== undefined) {
      window.clearInterval(typingTimerRef.current);
    }

    const fullContent = message.content;
    let visibleLength = 0;
    // 응답 전문은 받은 뒤에도 한 번에 붙이지 않고 조금씩 늘려 스트리밍처럼 보이게 한다.
    setMessages((currentMessages) => [
      ...currentMessages.filter((item) => item.id !== message.id),
      withTypingContent(message, ""),
    ]);

    typingTimerRef.current = window.setInterval(() => {
      visibleLength = Math.min(fullContent.length, visibleLength + ASSISTANT_TYPING_CHUNK_SIZE);
      const nextContent = fullContent.slice(0, visibleLength);
      const done = visibleLength >= fullContent.length;
      setMessages((currentMessages) => currentMessages.map((item) => (
        item.id === message.id
          ? (done ? { ...message, content: nextContent } : withTypingContent(message, nextContent))
          : item
      )));
      if (done && typingTimerRef.current !== undefined) {
        window.clearInterval(typingTimerRef.current);
        typingTimerRef.current = undefined;
      }
    }, ASSISTANT_TYPING_INTERVAL_MS);
  }, []);

  const loadMessages = useCallback(async (conversation: AssistantConversation) => {
    deliveryRequestTokenRef.current += 1;
    const requestId = ++loadMessagesRequestRef.current;
    setError(undefined);
    activeConversationIdRef.current = conversation.id;
    setActiveConversation(conversation);
    setUnknownDelivery(undefined);
    try {
      const response = await getAssistantMessages(conversation.id);
      if (requestId !== loadMessagesRequestRef.current) return [];

      activeConversationIdRef.current = response.conversation.id;
      setActiveConversation(response.conversation);
      setMessages(response.messages);
      setClientUpgradeRequired(false);
      setOpenEvidenceMessageIds(new Set());
      upsertConversationByUpdatedAt(response.conversation);
      return response.messages;
    } catch (caught) {
      if (requestId === loadMessagesRequestRef.current) {
        if (isClientUpgradeError(caught)) {
          setClientUpgradeRequired(true);
          setError("새 버전이 필요합니다. 페이지를 새로고침해 주세요.");
        } else {
          setError(caught instanceof ApiError ? caught.message : "대화 내용을 불러오지 못했습니다.");
        }
      }
      return [];
    }
  }, [upsertConversationByUpdatedAt]);

  const loadConversations = useCallback(async () => {
    setBooting(true);
    setError(undefined);
    try {
      const response = await listAssistantConversations();
      const sortedConversations = sortConversationsByUpdatedAt(response.conversations);
      setConversations(sortedConversations);
      if (sortedConversations.length > 0) {
        await loadMessages(sortedConversations[0]);
      } else {
        await createConversation(DEFAULT_CONVERSATION_TITLE);
      }
    } catch (caught) {
      if (isClientUpgradeError(caught)) {
        setClientUpgradeRequired(true);
        setError("새 버전이 필요합니다. 페이지를 새로고침해 주세요.");
      } else {
        setError(caught instanceof ApiError ? caught.message : "AI 비서 대화를 준비하지 못했습니다.");
      }
    } finally {
      setBooting(false);
    }
  }, [createConversation, loadMessages]);

  const reconcileDelivery = useCallback(async (
    owner: DeliveryOwner,
    failureStatus: number,
    failureCopy: string,
  ) => {
    if (!isCurrentDelivery(
      owner,
      deliveryRequestTokenRef.current,
      activeConversationIdRef.current,
    )) return;

    try {
      const response = await getAssistantMessages(owner.conversationId);
      if (!isCurrentDelivery(
        owner,
        deliveryRequestTokenRef.current,
        activeConversationIdRef.current,
      )) return;

      activeConversationIdRef.current = response.conversation.id;
      setActiveConversation(response.conversation);
      setMessages(response.messages);
      setOpenEvidenceMessageIds(new Set());
      setUnknownDelivery(undefined);
      upsertConversationByUpdatedAt(response.conversation);
      setError(failureStatus === 502
        ? "답변을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요."
        : undefined);
    } catch {
      if (!isCurrentDelivery(
        owner,
        deliveryRequestTokenRef.current,
        activeConversationIdRef.current,
      )) return;
      setMessages((currentMessages) => markOwnedOptimisticUnknown(currentMessages, owner));
      setUnknownDelivery({ owner, failureStatus, failureCopy });
      setError(failureCopy);
    }
  }, [upsertConversationByUpdatedAt]);

  const sendMessage = useCallback(async (content: string) => {
    const trimmedContent = content.trim();
    if (
      !trimmedContent
      || loadingRef.current
      || unknownDelivery !== undefined
      || clientUpgradeRequired
    ) return;

    loadingRef.current = true;
    setLoading(true);
    setError(undefined);
    setQuery("");
    let owner: DeliveryOwner | undefined;
    try {
      const conversation = activeConversation ?? await createConversation(trimmedContent);
      const requestToken = ++deliveryRequestTokenRef.current;
      const optimisticMessage = createOptimisticUserMessage(
        conversation.id,
        trimmedContent,
        nextOptimisticMessageIdRef.current--,
        requestToken,
      );
      owner = {
        requestToken,
        conversationId: conversation.id,
        optimisticMessageId: optimisticMessage.id,
      };
      // 사용자가 보낸 말은 서버 응답을 기다리지 않고 바로 대화창에 올린다.
      setMessages((currentMessages) => [...currentMessages, optimisticMessage]);
      const response = await createAssistantMessage(conversation.id, trimmedContent);
      if (!isCurrentDelivery(
        owner,
        deliveryRequestTokenRef.current,
        activeConversationIdRef.current,
      )) return;

      activeConversationIdRef.current = response.conversation.id;
      setActiveConversation(response.conversation);
      setMessages((currentMessages) => replaceOptimisticMessage(
        currentMessages,
        optimisticMessage.id,
        markDelivery(response.user_message, "persisted", requestToken),
      ));
      setUnknownDelivery(undefined);
      setOpenEvidenceMessageIds(new Set());
      upsertConversationByUpdatedAt(response.conversation);
      revealAssistantMessage(response.assistant_message);
    } catch (caught) {
      if (
        owner === undefined
        || !isCurrentDelivery(
          owner,
          deliveryRequestTokenRef.current,
          activeConversationIdRef.current,
        )
      ) {
        if (owner === undefined) {
          if (isClientUpgradeError(caught)) {
            setClientUpgradeRequired(true);
            setError("새 버전이 필요합니다. 페이지를 새로고침해 주세요.");
          } else {
            setError(caught instanceof ApiError ? caught.message : "메시지를 보내지 못했습니다.");
          }
        }
        return;
      }
      const caughtOwner = owner;

      if (caught instanceof ApiError && caught.status === 409 && caught.code === "client_upgrade_required") {
        setMessages((currentMessages) => removeOwnedOptimistic(currentMessages, caughtOwner));
        setClientUpgradeRequired(true);
        setError("새 버전이 필요합니다. 페이지를 새로고침해 주세요.");
      } else if (caught instanceof ApiError && [403, 404, 422].includes(caught.status)) {
        setMessages((currentMessages) => removeOwnedOptimistic(currentMessages, caughtOwner));
        setError(definiteRefusalCopy(caught));
      } else {
        const failureStatus = caught instanceof ApiError ? caught.status : 500;
        const failureCopy = caught instanceof ApiError
          && caught.status === 409
          && caught.code === "budget_exceeded"
          ? caught.message
          : deliveryFailureCopy(failureStatus);
        await reconcileDelivery(caughtOwner, failureStatus, failureCopy);
      }
    } finally {
      loadingRef.current = false;
      setLoading(false);
    }
  }, [
    activeConversation,
    clientUpgradeRequired,
    createConversation,
    reconcileDelivery,
    revealAssistantMessage,
    unknownDelivery,
    upsertConversationByUpdatedAt,
  ]);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (loading) return;
    void sendMessage(query);
  }

  async function handleNewConversation() {
    if (creatingConversationRef.current || unknownDelivery !== undefined || clientUpgradeRequired) return;
    creatingConversationRef.current = true;
    try {
      setError(undefined);
      setQuery("");
      if (isReusableActiveConversation(activeConversation, messages)) return;

      const reusableConversation = conversations.find(isDefaultConversation);
      if (reusableConversation) {
        await loadMessages(reusableConversation);
        return;
      }

      await createConversation(DEFAULT_CONVERSATION_TITLE);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "새 대화를 만들지 못했습니다.");
    } finally {
      creatingConversationRef.current = false;
    }
  }

  function toggleEvidence(messageId: number) {
    setOpenEvidenceMessageIds((current) => {
      const next = new Set(current);
      if (next.has(messageId)) {
        next.delete(messageId);
      } else {
        next.add(messageId);
      }
      return next;
    });
  }

  async function copyMessage(message: AssistantMessage) {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopiedMessageId(message.id);
      window.setTimeout(() => setCopiedMessageId((current) => current === message.id ? undefined : current), 1600);
    } catch {
      setError("메시지를 복사하지 못했습니다.");
    }
  }

  async function approveEmailDraft(messageId: number) {
    setSendingEmailMessageId(messageId);
    setError(undefined);
    try {
      const response = await sendAssistantEmailDraft(messageId);
      setMessages((currentMessages) => currentMessages.map((message) => (
        message.id === messageId ? response.message : message
      )));
    } catch (caught) {
      if (isClientUpgradeError(caught)) {
        setClientUpgradeRequired(true);
        setError("새 버전이 필요합니다. 페이지를 새로고침해 주세요.");
      } else {
        setError(caught instanceof ApiError ? caught.message : "메일을 보내지 못했습니다.");
      }
    } finally {
      setSendingEmailMessageId(undefined);
    }
  }

  async function retryUnknownDelivery() {
    if (unknownDelivery === undefined || loadingRef.current) return;
    loadingRef.current = true;
    setLoading(true);
    await reconcileDelivery(
      unknownDelivery.owner,
      unknownDelivery.failureStatus,
      unknownDelivery.failureCopy,
    );
    loadingRef.current = false;
    setLoading(false);
  }

  useEffect(() => {
    if (handoffConsumedRef.current) return;
    handoffConsumedRef.current = true;
    if (searchParams.has("q")) {
      ephemeralSearchHandoff.consume();
      setQuery("");
      router.replace("/search");
      return;
    }
    setQuery(ephemeralSearchHandoff.consume() ?? "");
  }, [router, searchParams]);

  useEffect(() => {
    void loadConversations();
  }, [loadConversations]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ block: "end" });
  }, [messages.length, loading]);

  useEffect(() => () => {
    if (typingTimerRef.current !== undefined) {
      window.clearInterval(typingTimerRef.current);
    }
  }, []);

  useEffect(() => {
    setHydrated(true);
  }, []);

  return (
    <div
      data-assistant-hydrated={hydrated ? "true" : "false"}
      className="reference-dashboard utility-workspace utility-workspace-chat h-[calc(100vh-7rem)] overflow-hidden"
    >
      <section
        className={`grid h-full min-h-0 transition-[grid-template-columns,gap] duration-300 ease-out ${
          sidebarCollapsed ? "gap-0 lg:grid-cols-[0_minmax(0,1fr)]" : "gap-3 lg:grid-cols-[280px_minmax(0,1fr)]"
        }`}
      >
        <aside
          aria-label="대화 목록"
          data-expanded={sidebarCollapsed ? "false" : "true"}
          className={`panel reference-panel group flex h-full min-h-0 flex-col overflow-hidden rounded-2xl transition-all duration-300 ease-out hover:shadow-panel-hover ${
            sidebarCollapsed ? "pointer-events-none w-0 p-0 opacity-0" : "w-full"
          }`}
        >
          <div className="flex items-center justify-between gap-2 border-b border-line pb-3">
            <div className="min-w-0">
              <p className="text-[12px] font-bold text-[var(--primary-dark)]">AI 비서</p>
              <h2 className="truncate text-[15px] font-extrabold">대화</h2>
            </div>
            <div className="flex shrink-0 gap-2">
              <button
                type="button"
                title="대화 목록 접기"
                aria-label="대화 목록 접기"
                data-assistant-history-collapse
                onClick={() => setSidebarCollapsed(true)}
                className="row-action h-9 w-9 p-0 transition-transform duration-200 group-hover:scale-105"
              >
                <PanelLeftClose className="h-4 w-4" aria-hidden="true" />
              </button>
              <button
                type="button"
                title="새 대화 만들기"
                aria-label="새 대화 만들기"
                onClick={() => void handleNewConversation()}
                className="row-action h-9 w-9 p-0 transition-transform duration-200 group-hover:scale-105"
                disabled={booting || unknownDelivery !== undefined || clientUpgradeRequired}
              >
                <Plus className="h-4 w-4" aria-hidden="true" />
              </button>
            </div>
          </div>

          <div aria-label="대화 히스토리" className="mt-3 flex-1 space-y-1 overflow-y-auto pr-1">
            {conversations.map((conversation) => {
              const selected = conversation.id === activeConversation?.id;
              return (
                <button
                  key={conversation.id}
                  type="button"
                  onClick={() => void loadMessages(conversation)}
                  disabled={unknownDelivery !== undefined || clientUpgradeRequired}
                  className={`w-full truncate rounded-xl px-3 py-2 text-left text-[13px] font-bold transition ${
                    selected
                      ? "bg-[var(--primary-soft)] text-[var(--primary-dark)]"
                      : "text-[var(--ink)] hover:bg-[var(--glass-strong)]"
                  } disabled:cursor-not-allowed disabled:opacity-50`}
                >
                  {conversation.title || DEFAULT_CONVERSATION_TITLE}
                </button>
              );
            })}
            {booting ? (
              <div className="rounded-xl border border-dashed border-line bg-surface-soft p-3 text-[13px] text-muted">
                대화를 불러오는 중입니다.
              </div>
            ) : null}
          </div>
        </aside>

        <main className="relative flex h-full min-h-0 flex-col overflow-hidden rounded-2xl bg-[var(--surface)] p-0 shadow-panel">
          {sidebarCollapsed ? (
            <button
              type="button"
              title="대화 목록 펼치기"
              aria-label="대화 목록 펼치기"
              data-assistant-history-open
              onClick={() => setSidebarCollapsed(false)}
              className="group absolute left-4 top-4 z-10 rounded-full outline-none transition-transform duration-200 hover:scale-105 focus-visible:ring-2 focus-visible:ring-[var(--primary)]"
            >
              <div className="grid h-10 w-10 place-items-center rounded-full bg-[var(--primary-soft)] text-[var(--primary-dark)] shadow-xs transition group-hover:bg-[var(--primary)] group-hover:text-white group-hover:shadow-panel-hover">
                <MessageSquareText className="h-4 w-4" aria-hidden="true" />
              </div>
            </button>
          ) : null}
          {!sidebarCollapsed ? (
            <button
              type="button"
              title="대화 목록 접기"
              aria-label="대화 목록 접기"
              data-assistant-history-collapse
              onClick={() => setSidebarCollapsed(true)}
              className="absolute left-4 top-4 z-10 inline-flex h-10 items-center gap-2 rounded-full border border-slate-200 bg-white/90 px-3 text-[12px] font-bold text-slate-600 shadow-xs backdrop-blur transition hover:border-indigo-200 hover:bg-indigo-50 hover:text-indigo-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--primary)]"
            >
              <PanelLeftClose className="h-4 w-4" aria-hidden="true" />
              <span className="hidden sm:inline">접기</span>
            </button>
          ) : null}
          {error ? (
            <div className="mx-auto mt-4 w-full max-w-3xl rounded-lg border border-red-200 bg-red-50 p-3 text-[13px] text-red-800">
              <p>{error}</p>
              {clientUpgradeRequired ? (
                <button
                  type="button"
                  onClick={() => window.location.reload()}
                  className="mt-2 rounded-full bg-red-700 px-3 py-2 text-[12px] font-bold text-white"
                >
                  페이지 새로고침
                </button>
              ) : null}
            </div>
          ) : null}

          <div className={`min-h-0 flex-1 overflow-y-auto px-4 pb-5 sm:px-6 ${sidebarCollapsed ? "pt-16" : "pt-5"}`}>
            <div className="mx-auto flex max-w-3xl flex-col gap-5">
              {messages.map((message) => (
                <AssistantBubble
                  key={message.id}
                  message={message}
                  evidenceOpen={openEvidenceMessageIds.has(message.id)}
                  onToggleEvidence={() => toggleEvidence(message.id)}
                  copied={copiedMessageId === message.id}
                  onCopy={() => void copyMessage(message)}
                  sendingEmail={sendingEmailMessageId === message.id}
                  emailActionBlocked={unknownDelivery !== undefined || clientUpgradeRequired}
                  onApproveEmail={() => void approveEmailDraft(message.id)}
                />
              ))}
              {!booting && messages.length === 0 ? (
                <div className="flex min-h-[420px] flex-col items-center justify-center text-center">
                  <Bot className="h-9 w-9 text-[var(--primary)]" aria-hidden="true" />
                  <h1 className="mt-4 text-[22px] font-black">무엇을 도와드릴까요?</h1>
                  <p className="mt-2 max-w-lg text-[13px] leading-6 text-muted">
                    회사 기억, 승인된 지식, 접근 가능한 근거를 바탕으로 답변합니다.
                  </p>
                </div>
              ) : null}
              {loading ? (
                <div className="flex justify-start">
                  <div className="rounded-lg bg-[var(--glass-elevated)] px-4 py-3 text-[13px] text-muted">
                    답변을 정리하는 중입니다...
                  </div>
                </div>
              ) : null}
              <div ref={messagesEndRef} />
            </div>
          </div>

          <form
            aria-label="AI 비서 입력"
            onSubmit={submit}
            className="bg-white/80 px-4 pb-4 pt-2 backdrop-blur-md sm:px-6"
          >
            <div className="mx-auto max-w-3xl">
              <div className="mb-3 flex flex-wrap justify-center gap-2">
                {SUGGESTED_QUESTIONS.map((question) => (
                  <button
                    key={question}
                    type="button"
                    onClick={() => void sendMessage(question)}
                    disabled={loading || booting || unknownDelivery !== undefined || clientUpgradeRequired}
                    className="rounded-full bg-[var(--primary)] px-3 py-2 text-[12px] font-bold leading-5 text-white transition hover:bg-[var(--primary-dark)] disabled:bg-neutral-300"
                  >
                    {question}
                  </button>
                ))}
              </div>
              <label htmlFor="assistant-query" className="sr-only">AI 비서에게 질문</label>
              <div className="flex items-end gap-2 rounded-2xl border border-line bg-[var(--glass-elevated)] p-2 shadow-xs focus-within:border-[var(--primary)]">
                <textarea
                  id="assistant-query"
                  aria-label="AI 비서에게 질문"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      void sendMessage(query);
                    }
                  }}
                  className="max-h-32 min-h-11 min-w-0 flex-1 resize-none bg-transparent px-2 py-2 text-[14px] leading-6 outline-none"
                  placeholder="회사 기억에서 무엇을 찾아볼까요?"
                  disabled={booting || unknownDelivery !== undefined || clientUpgradeRequired}
                  rows={1}
                />
                <button
                  type="submit"
                  aria-label="전송"
                  disabled={
                    loading
                    || booting
                    || unknownDelivery !== undefined
                    || clientUpgradeRequired
                    || query.trim().length === 0
                  }
                  className="row-action h-10 w-10 shrink-0 p-0 disabled:bg-neutral-300"
                >
                  <Send className="h-4 w-4" aria-hidden="true" />
                </button>
              </div>
              {unknownDelivery ? (
                <div className="mt-2 flex items-center justify-between gap-3 rounded-lg bg-amber-50 px-3 py-2 text-[12px] text-amber-900">
                  <span>전송 상태를 확인하지 못했습니다.</span>
                  <button
                    type="button"
                    onClick={() => void retryUnknownDelivery()}
                    disabled={loading}
                    className="shrink-0 rounded-full bg-amber-800 px-3 py-1.5 font-bold text-white disabled:bg-neutral-300"
                  >
                    상태 다시 확인
                  </button>
                </div>
              ) : null}
            </div>
          </form>
        </main>
      </section>
    </div>
  );
}

function SearchPageFallback() {
  return <div className="panel reference-panel utility-workspace p-8 text-[13px] text-muted">AI 비서 화면을 준비하고 있습니다.</div>;
}

function AssistantBubble({
  message,
  evidenceOpen,
  onToggleEvidence,
  copied,
  onCopy,
  sendingEmail,
  emailActionBlocked,
  onApproveEmail,
}: {
  message: AssistantMessage;
  evidenceOpen: boolean;
  onToggleEvidence: () => void;
  copied: boolean;
  onCopy: () => void;
  sendingEmail: boolean;
  emailActionBlocked: boolean;
  onApproveEmail: () => void;
}) {
  const isAssistant = message.role === "assistant";
  const evidenceCount = evidenceItemCount(message);
  const isTyping = message.metadata?.ui_status === "typing";
  const emailDraft = getEmailDraftView(message);
  const plainTextRag = isPlainTextRagMessage(message.metadata);
  const permissionNotice = plainTextRag ? publicPermissionNotice(message) : null;
  const deliveryStatus = getDeliveryStatus(message);

  return (
    <article className={`flex ${isAssistant ? "justify-start" : "justify-end"}`}>
      <div className={`min-w-0 ${isAssistant ? "w-full" : "max-w-[82%]"}`}>
        <div
          className={
            isAssistant
              ? "rounded-lg bg-transparent px-4 py-3 text-[14px] leading-7"
              : "rounded-full bg-[var(--primary)] px-3 py-2 text-[12px] font-bold leading-5 text-white"
          }
        >
          {isAssistant ? (
            <>
              {plainTextRag ? (
                <p
                  data-testid="rag-plain-text-answer"
                  className="whitespace-pre-wrap [overflow-wrap:anywhere]"
                >
                  {message.content}
                </p>
              ) : (
                <MarkdownContent content={message.content} />
              )}
              {isTyping ? <span className="ml-0.5 animate-pulse text-[var(--primary)]">▍</span> : null}
            </>
          ) : (
            <p className="whitespace-pre-wrap break-words">{message.content}</p>
          )}
        </div>

        {isAssistant && permissionNotice ? (
          <div className="mx-4 flex items-start gap-2 rounded-md border border-amber-200 bg-amber-50 p-3 text-[12px] leading-5 text-amber-900">
            <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            <p>{permissionNotice}</p>
          </div>
        ) : null}

        {!isAssistant && deliveryStatus === "unknown" ? (
          <p className="mt-1 text-right text-[11px] font-bold text-amber-800">
            전송 상태를 확인하지 못했습니다.
          </p>
        ) : null}

        {isAssistant && emailDraft ? (
          <EmailDraftActions
            status={emailDraft.status}
            sending={sendingEmail}
            blocked={emailActionBlocked}
            onApprove={onApproveEmail}
          />
        ) : null}

        <div className={`mt-1 flex ${isAssistant ? "justify-start" : "justify-end"}`}>
          <button
            type="button"
            onClick={onCopy}
            className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[12px] font-bold text-muted hover:bg-[var(--glass-strong)] hover:text-[var(--ink)]"
          >
            <Copy className="h-3.5 w-3.5" aria-hidden="true" />
            {copied ? "복사됨" : "복사"}
          </button>
        </div>

        {isAssistant && evidenceCount > 0 ? (
          <div className="mt-2">
            <button
              type="button"
              onClick={onToggleEvidence}
              className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[12px] font-bold text-[var(--primary-dark)] hover:bg-[var(--primary-soft)]"
            >
              {evidenceOpen ? (
                <ChevronDown className="h-3.5 w-3.5" aria-hidden="true" />
              ) : (
                <ChevronRight className="h-3.5 w-3.5" aria-hidden="true" />
              )}
              근거와 출처 {evidenceCount.toLocaleString()}개 {evidenceOpen ? "접기" : "펼치기"}
            </button>
            {evidenceOpen ? (
              <EvidenceDisclosure
                message={message}
                showPermissionNotice={!plainTextRag}
                showLegacySourceLinks={!plainTextRag}
              />
            ) : null}
          </div>
        ) : null}
      </div>
    </article>
  );
}

function EmailDraftActions({
  status,
  sending,
  blocked,
  onApprove,
}: {
  status: string;
  sending: boolean;
  blocked: boolean;
  onApprove: () => void;
}) {
  if (status === "sent") {
    return <p className="mt-2 text-[12px] font-bold text-emerald-700">전송 완료</p>;
  }

  return (
    <div className="mt-2 flex flex-wrap items-center gap-2 rounded-xl bg-[var(--glass-elevated)] p-3">
      <p className="text-[12px] font-bold text-muted">승인 전까지 메일은 전송되지 않습니다.</p>
      <button
        type="button"
        onClick={onApprove}
        disabled={sending || blocked}
        className="rounded-full bg-[var(--primary)] px-3 py-2 text-[12px] font-bold leading-5 text-white transition hover:bg-[var(--primary-dark)] disabled:bg-neutral-300"
      >
        {sending ? "전송 중" : "승인하고 보내기"}
      </button>
    </div>
  );
}

function MarkdownContent({ content }: { content: string }) {
  const lines = content.split(/\r?\n/);
  const blocks = [];
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (!line.trim()) {
      blocks.push(<div key={`space-${index}`} className="h-2" />);
      continue;
    }

    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length;
      const className = level === 1
        ? "text-[18px] font-black"
        : level === 2
          ? "text-[16px] font-extrabold"
          : "text-[14px] font-extrabold";
      blocks.push(<h3 key={`heading-${index}`} className={`mt-2 break-words ${className}`}>{renderInlineMarkdown(heading[2])}</h3>);
      continue;
    }

    if (line.trimStart().startsWith("- ")) {
      const items = [];
      let listIndex = index;
      while (listIndex < lines.length && lines[listIndex].trimStart().startsWith("- ")) {
        items.push(lines[listIndex].trimStart().slice(2));
        listIndex += 1;
      }
      blocks.push(
        <ul key={`list-${index}`} className="my-2 list-disc space-y-1 pl-5">
          {items.map((item, itemIndex) => (
            <li key={`${item}-${itemIndex}`} className="break-words">{renderInlineMarkdown(item)}</li>
          ))}
        </ul>,
      );
      index = listIndex - 1;
      continue;
    }

    blocks.push(<p key={`paragraph-${index}`} className="whitespace-pre-wrap break-words">{renderInlineMarkdown(line)}</p>);
  }

  return <div className="space-y-1">{blocks}</div>;
}

function renderInlineMarkdown(text: string) {
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g;
  const parts = text.split(pattern);
  return parts.map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={`${part}-${index}`}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={`${part}-${index}`} className="rounded bg-surface-soft px-1 py-0.5 text-[13px]">{part.slice(1, -1)}</code>;
    }
    const link = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (link) {
      return (
        <a
          key={`${part}-${index}`}
          href={link[2]}
          target="_blank"
          rel="noreferrer"
          className="font-bold text-[var(--primary-dark)] underline-offset-4 hover:underline"
        >
          {link[1]}
        </a>
      );
    }
    return part;
  });
}

function EvidenceDisclosure({
  message,
  showPermissionNotice,
  showLegacySourceLinks,
}: {
  message: AssistantMessage;
  showPermissionNotice: boolean;
  showLegacySourceLinks: boolean;
}) {
  const citations = message.citations ?? [];
  const citationSnippets = new Set(citations.map((citation) => citation.source_snippet));
  const snippets = (message.source_snippets ?? []).filter((snippet) => !citationSnippets.has(snippet));
  const links = showLegacySourceLinks ? (message.source_links ?? []) : [];

  return (
    <div className="mt-2 max-h-72 overflow-y-auto rounded-lg border border-line bg-[var(--glass-elevated)] p-3">
      {showPermissionNotice && message.permission_notice ? (
        <div className="mb-3 flex items-start gap-2 rounded-md border border-amber-200 bg-amber-50 p-3 text-[12px] leading-5 text-amber-900">
          <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
          <p>{message.permission_notice}</p>
        </div>
      ) : null}

      <EvidenceCitationList citations={citations} />

      {snippets.length > 0 ? (
        <section className="mt-3">
          <h3 className="mb-2 flex items-center gap-1 text-[12px] font-extrabold">
            <FileText className="h-3.5 w-3.5" aria-hidden="true" />
            원문 발췌
          </h3>
          <div className="space-y-2">
            {snippets.map((snippet, index) => (
              <p
                key={`${snippet}-${index}`}
                className="rounded-md border border-line bg-surface-soft p-3 text-[12px] leading-5 text-muted"
              >
                {snippet}
              </p>
            ))}
          </div>
        </section>
      ) : null}

      {links.length > 0 ? (
        <section className="mt-3">
          <h3 className="mb-2 flex items-center gap-1 text-[12px] font-extrabold">
            <Link2 className="h-3.5 w-3.5" aria-hidden="true" />
            출처 링크
          </h3>
          <div className="space-y-2">
            {links.map((link, index) => (
              <a
                key={`${link}-${index}`}
                href={link}
                target="_blank"
                rel="noreferrer"
                className="block break-all rounded-md border border-line bg-surface-soft p-3 text-[12px] font-bold text-[var(--primary-dark)] underline-offset-4 hover:underline"
              >
                {link}
              </a>
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}

function EvidenceCitationList({ citations }: { citations: RagCitation[] }) {
  if (citations.length === 0) return null;

  return (
    <section>
      <h3 className="mb-2 flex items-center gap-1 text-[12px] font-extrabold">
        <Sparkles className="h-3.5 w-3.5" aria-hidden="true" />
        인용 근거
      </h3>
      <div className="space-y-2">
        {citations.map((citation, index) => {
          const content = (
            <>
            <span className="font-bold text-[var(--primary-dark)]">근거 {index + 1}</span>
            <span className="mt-1 block break-all text-muted">{citation.source_id}</span>
            <span className="mt-1 block text-muted">관련도 {citation.relevance_score.toFixed(2)}</span>
            {citation.matched_terms.length ? (
              <span className="mt-2 flex flex-wrap gap-1">
                {citation.matched_terms.map((term, termIndex) => (
                  <span key={`${term}-${termIndex}`} className="filter-pill active">{term}</span>
                ))}
              </span>
            ) : null}
            <span className="mt-2 block border-l-2 border-line pl-3 leading-5 text-muted">
              {citation.source_snippet}
            </span>
            </>
          );
          return isSafeCitationUrl(citation.source_url) ? (
            <a
              key={`${citation.source_id}-${index}`}
              href={citation.source_url}
              target="_blank"
              rel="noopener noreferrer"
              className="block rounded-md border border-line bg-surface-soft p-3 text-[12px] hover:bg-[var(--glass-strong)]"
            >
              {content}
            </a>
          ) : (
            <div
              key={`${citation.source_id}-${index}`}
              className="block rounded-md border border-line bg-surface-soft p-3 text-[12px]"
            >
              {content}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function evidenceItemCount(message: AssistantMessage) {
  if (isPlainTextRagMessage(message.metadata)) {
    return Math.max(message.citations.length, message.source_snippets.length);
  }
  return Math.max(
    message.citations.length,
    message.source_links.length,
    message.source_snippets.length,
  );
}

function isDefaultConversation(conversation: AssistantConversation) {
  return conversation.title.trim() === DEFAULT_CONVERSATION_TITLE;
}

function createOptimisticUserMessage(
  conversationId: number,
  content: string,
  id: number,
  requestToken: number,
): AssistantMessage {
  return markDelivery({
    id,
    conversation_id: conversationId,
    role: "user",
    content,
    citations: [],
    source_ids: [],
    source_links: [],
    source_snippets: [],
    permission_level: null,
    hidden_match_count: 0,
    permission_notice: null,
    agent_run_id: null,
    metadata: { ui_status: "optimistic" },
    created_at: new Date().toISOString(),
  }, "sending", requestToken);
}

function replaceOptimisticMessage(
  messages: AssistantMessage[],
  optimisticMessageId: number,
  persistedMessage: AssistantMessage,
) {
  let replaced = false;
  const nextMessages = messages.map((message) => {
    if (message.id !== optimisticMessageId) return message;
    replaced = true;
    return persistedMessage;
  });
  return replaced ? nextMessages : [...nextMessages, persistedMessage];
}

function withTypingContent(message: AssistantMessage, content: string): AssistantMessage {
  return {
    ...message,
    content,
    metadata: {
      ...message.metadata,
      ui_status: "typing",
    },
  };
}

function getEmailDraftView(message: AssistantMessage): { status: string } | undefined {
  if (message.metadata?.action_type !== "email_draft") return undefined;
  return {
    status: typeof message.metadata.status === "string" ? message.metadata.status : "pending_approval",
  };
}

function isReusableActiveConversation(
  conversation: AssistantConversation | undefined,
  messages: AssistantMessage[],
) {
  return Boolean(conversation && isDefaultConversation(conversation) && messages.length === 0);
}

function sortConversationsByUpdatedAt(conversations: AssistantConversation[]) {
  return [...conversations].sort((left, right) => {
    const timeDiff = new Date(right.updated_at).getTime() - new Date(left.updated_at).getTime();
    return timeDiff || right.id - left.id;
  });
}

function deliveryFailureCopy(status: number): string {
  return status === 502
    ? "답변을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요."
    : "요청 처리 상태를 확인하지 못했습니다. 잠시 후 다시 시도해 주세요.";
}

function definiteRefusalCopy(error: ApiError): string {
  if (error.status === 404) {
    return "대화를 찾을 수 없습니다. 대화 목록을 새로고침해 주세요.";
  }
  if (error.status === 422 && error.code === null) {
    return "요청 내용을 확인해 주세요.";
  }
  return error.message;
}

function isClientUpgradeError(error: unknown): error is ApiError {
  return error instanceof ApiError
    && error.status === 409
    && error.code === "client_upgrade_required";
}
