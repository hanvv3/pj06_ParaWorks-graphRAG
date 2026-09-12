import type {
  AskResponse,
  AssistantConversation,
  AssistantMessage,
  RagCitation,
  SearchResult,
} from "../src/lib/api/types";

type Expect<T extends true> = T;
type IsRequired<T, K extends keyof T> = object extends Pick<T, K> ? false : true;
type IsAny<T> = 0 extends (1 & T) ? true : false;
type IsExact<T, Expected> =
  IsAny<T> extends true
    ? false
    : IsAny<Expected> extends true
      ? false
      : [T] extends [Expected]
        ? [Expected] extends [T]
          ? true
          : false
        : false;
type IsRequiredExact<T, K extends keyof T, Expected> =
  IsRequired<T, K> extends true ? IsExact<T[K], Expected> : false;
type IsRequiredNonNullString<T, K extends keyof T> =
  IsRequiredExact<T, K, string>;

export type RagV2WireContractAssertions = [
  Expect<IsRequiredExact<AskResponse, "permission_level", string | null>>,
  Expect<IsRequiredExact<AskResponse, "permission_notice", string | null>>,
  Expect<IsRequiredExact<AskResponse, "agent_run_id", number | null>>,
  Expect<IsRequiredExact<AssistantConversation, "summary", string | null>>,
  Expect<IsRequiredExact<AssistantMessage, "permission_level", string | null>>,
  Expect<IsRequiredExact<AssistantMessage, "permission_notice", string | null>>,
  Expect<IsRequiredExact<AssistantMessage, "agent_run_id", number | null>>,
  Expect<IsRequiredExact<SearchResult, "source_type", string | null>>,
  Expect<IsRequiredExact<SearchResult, "parser_status", string | null>>,
  Expect<IsRequiredExact<SearchResult, "parser_status_reason", string | null>>,
  Expect<IsRequiredExact<SearchResult, "revision_id", string | null>>,
  Expect<IsRequiredExact<RagCitation, "source_type", string | null>>,
  Expect<IsRequiredNonNullString<SearchResult, "source_url">>,
  Expect<IsRequiredNonNullString<RagCitation, "source_url">>,
];

// These negative fixtures must remain compile errors. An unused directive means
// the predicate accepted a wire type that is broader or narrower than backend.
// @ts-expect-error permission_level must reject a numeric non-null branch
export type RejectNumericPermission = Expect<IsRequiredExact<
  { permission_level: number | null },
  "permission_level",
  string | null
>>;
// @ts-expect-error permission_notice must reject unknown
export type RejectUnknownNotice = Expect<IsRequiredExact<
  { permission_notice: unknown },
  "permission_notice",
  string | null
>>;
// @ts-expect-error agent_run_id must reject a string non-null branch
export type RejectStringRunId = Expect<IsRequiredExact<
  { agent_run_id: string | null },
  "agent_run_id",
  number | null
>>;
// @ts-expect-error any must never satisfy an exact nullable wire type
export type RejectAnyPermission = Expect<IsRequiredExact<
  { permission_level: ReturnType<typeof JSON.parse> },
  "permission_level",
  string | null
>>;
// @ts-expect-error source_url must be exactly string, not a literal-only subtype
export type RejectNarrowUrl = Expect<IsRequiredNonNullString<
  { source_url: "https://only.example" },
  "source_url"
>>;
