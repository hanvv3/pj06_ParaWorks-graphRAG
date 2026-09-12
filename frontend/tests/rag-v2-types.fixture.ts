import type {
  AskResponse,
  AssistantConversation,
  AssistantMessage,
  RagCitation,
  SearchResult,
} from "../src/lib/api/types";

type Expect<T extends true> = T;
type IsRequired<T, K extends keyof T> = object extends Pick<T, K> ? false : true;
type IsNullable<T, K extends keyof T> = null extends T[K] ? true : false;
type IsRequiredNullable<T, K extends keyof T> =
  IsRequired<T, K> extends true ? IsNullable<T, K> : false;
type IsRequiredNonNullString<T, K extends keyof T> =
  IsRequired<T, K> extends true
    ? null extends T[K]
      ? false
      : undefined extends T[K]
        ? false
        : T[K] extends string
          ? true
          : false
    : false;

export type RagV2WireContractAssertions = [
  Expect<IsRequiredNullable<AskResponse, "permission_level">>,
  Expect<IsRequiredNullable<AskResponse, "permission_notice">>,
  Expect<IsRequiredNullable<AskResponse, "agent_run_id">>,
  Expect<IsRequiredNullable<AssistantConversation, "summary">>,
  Expect<IsRequiredNullable<AssistantMessage, "permission_level">>,
  Expect<IsRequiredNullable<AssistantMessage, "permission_notice">>,
  Expect<IsRequiredNullable<AssistantMessage, "agent_run_id">>,
  Expect<IsRequiredNullable<SearchResult, "source_type">>,
  Expect<IsRequiredNullable<SearchResult, "parser_status">>,
  Expect<IsRequiredNullable<SearchResult, "parser_status_reason">>,
  Expect<IsRequiredNullable<SearchResult, "revision_id">>,
  Expect<IsRequiredNullable<RagCitation, "source_type">>,
  Expect<IsRequiredNonNullString<SearchResult, "source_url">>,
  Expect<IsRequiredNonNullString<RagCitation, "source_url">>,
];
