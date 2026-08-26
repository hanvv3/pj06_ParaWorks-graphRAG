# LangChain·LangGraph Runtime Foundation 설계

검토 버전: 3
최종 재검토일: 2026-08-26

## 1. 결정 요약

ParaWorks는 Neo4j GraphRAG를 추가하기 전에 LangChain·LangGraph runtime 경계를
고도화한다. 현재 코드는 실제 LangChain과 LangGraph API를 호출하지만, LangGraph는
아직 직선형 callable wrapper이고 Review Queue checkpoint는 설명용 dictionary다.

이번 재검토에서 다음 결정을 확정한다.

1. 전체 작업을 하나의 implementation plan으로 진행하지 않는다.
2. dependency compatibility, runtime/checkpoint 기반, 실제 Review HITL, RAG graph
   전환을 각각 독립적인 green deliverable로 분리한다.
3. 기존 `/company-memory/run` route와 response shape는 즉시 제거하지 않고 V2
   endpoint를 별도로 둔다. 단, 실제 checkpoint가 아닌 legacy status 값은
   `metadata_only`로 정정한다.
4. checkpoint에는 질문·원문·snippet·URL·모델 출력과 같은 민감한 내용을 저장하지
   않는다.
5. ReviewItem, AgentRun, 승인 knowledge promotion은 cache가 아니라 row lock, DB
   unique constraint, workflow thread idempotency로 exactly-once 보호한다.
6. RAG V2는 모델이 서버 발급 evidence slot을 claim별로 선택하는 structured output을
   검증한 뒤, 서버가 선택된 canonical evidence만 기존 citation shape로 투영한다.
7. graph version은 application registry가 선택한다. LangGraph의 `checkpoint_ns`를
   application version namespace로 오용하지 않는다.
8. 실제 query classification과 Neo4j branch는 GraphRAG 단계에서 추가한다. 기반
   리팩터링에서는 keyword/pgvector conditional routing만 구현한다.

## 2. 현재 구조와 확인된 간극

현재 저장소는 다음 실제 library API를 사용한다.

- `langchain.agents.create_agent`
- `ChatOpenAI`, `ChatGoogleGenerativeAI`
- `with_structured_output(...).invoke(...)`
- `langgraph.graph.StateGraph`
- compiled graph의 `invoke()`와 Mermaid topology

하지만 현재 `backend/app/agent_runtime/orchestration.py`는 전달받은 Python callable을
정해진 순서대로 실행한다. 다음 LangGraph 기능은 사용하지 않는다.

- conditional edge
- checkpointer를 전달한 `compile()`
- `interrupt()`
- `Command(resume=...)`
- 안정적인 graph `thread_id`
- deploy 이후 재개를 위한 graph version

또한 현재 `company_memory` workflow는 다음 두 책임을 한 run에 결합한다.

- source agent를 실행해 Review Queue 후보 생성
- 같은 질문으로 RAG 답변과 AgentRun 생성

RAG 경로도 `/ask`, `/search`, assistant, company-memory가 직접 keyword/pgvector
service와 adapter를 조립한다. 이 상태에서 Neo4j를 먼저 붙이면 같은 retrieval
integration을 다시 옮겨야 한다.

## 3. 범위와 전달 단위

이 문서는 program-level design이다. 실행은 아래 네 개의 별도 implementation plan과
검증 checkpoint로 분리한다. 앞 단계가 green이 아니면 다음 단계로 넘어가지 않는다.

### Deliverable A — Dependency Compatibility

- 검증된 최신 minor line으로 dependency와 `uv.lock` 갱신
- 기존 provider adapter와 structured output compatibility test
- LangGraph graph/interrupt/checkpointer import smoke test
- production behavior 변경 없음

### Deliverable B — Runtime and Checkpoint Primitives

- JSON-safe graph state와 runtime context 계약
- `AgentWorkflowThread`, canonical evidence reference와 thread-bound idempotency schema
- checkpointer lifecycle, bootstrap, readiness, serializer 보안
- graph version registry와 feature flag mode matrix
- 아직 public route cutover 없음

### Deliverable C — Actual Review Queue HITL V2

- 기존 route와 분리된 V2 review-only workflow
- 실제 `interrupt()`와 동일 thread의 `Command(resume=...)`
- ReviewItem/AgentRun idempotent transaction, exactly-once promotion과 failure reconciliation
- Review Queue status state machine과 resume authorization
- 기존 `/ask`, `/search`, assistant는 변경하지 않음

### Deliverable D — Retriever Port and RAG Answer Graph V2

- keyword와 pgvector를 공통 retrieval port로 adapter화
- 실제 conditional edge로 backend routing
- permission defense-in-depth, bounded hidden-match, fallback trace
- shadow parity 후 `/ask`, `/search`, assistant를 순차 전환
- structured claim-to-evidence validation과 기존 citation API shape 유지

### 이번 program에서 제외

- Neo4j container, schema, constraint, vector index
- `neo4j-graphrag` 설치와 retriever
- PostgreSQL-to-Neo4j outbox projection
- Text2Cypher
- Knowledge Map UI 변경
- pgvector 제거
- live LLM, embedding, Slack, Gmail, Drive API를 호출하는 자동 테스트

## 4. Dependency 정책

2026-08-26 공식 PyPI project page를 다시 확인한 초기 target은 다음과 같다.

- `langchain>=1.3.17,<1.4.0`
- `langgraph>=1.2.11,<1.3.0`
- `langchain-openai>=1.6.0,<1.7.0`
- `langchain-google-genai>=4.3.5,<4.4.0`
- `langgraph-checkpoint-postgres>=3.1.2,<3.2.0`
- `psycopg[binary,pool]>=3.3.2`

`pyproject.toml`은 검증한 minor line을 제한하고 재현 가능한 exact resolution은
`uv.lock`에 둔다. upgrade는 관련 package만 대상으로 하며 unrelated dependency의
임의 broad upgrade를 하지 않는다.

dependency-only commit 전에 다음을 수행한다.

1. 공식 PyPI에서 target release 존재를 다시 기록한다.
2. `uv lock` resolver 결과와 transitive `langchain-core`, checkpoint package를
   확인한다.
3. provider adapter, `create_agent`, `with_structured_output`, `StateGraph`,
   `interrupt`, `Command`, `PostgresSaver` smoke test를 실행한다.
4. resolver/API incompatibility가 있으면 compatibility shim을 만들지 않고 마지막
   green minor 조합으로 낮추며 이유를 기록한다.

## 5. 전체 runtime 구조

### 5.1 Company Memory Review Graph V2

```text
START
  -> validate_input
  -> collect_evidence_refs
  -> plan_agent_runs
  -> draft_review_candidates_transaction
  -> route_review_boundary
       -> no_candidates -> finalize_no_candidates -> END
       -> candidates_created -> await_human_review [interrupt]
  -> verify_review_resolution_from_postgres
       -> unresolved -> await_human_review
       -> needs_more_evidence -> finalize_needs_more_evidence -> END
       -> approved_or_rejected -> finalize_review_trace -> END
```

이 graph는 후보 생성과 human review 조정만 담당한다. RAG 답변을 생성하지 않고
Review Queue 승인 transaction을 대체하지 않는다. 승인된 knowledge promotion은 기존
Review API와 PostgreSQL이 계속 source of truth다.

`await_human_review`의 resume payload는 실행 재개 acknowledgement일 뿐 승인 값이
아니다. `verify_review_resolution_from_postgres`가 현재 ReviewItem 상태와 권한을 다시
읽어 최종 판단한다.

### 5.2 RAG Answer Graph V2

```text
START
  -> validate_input
  -> resolve_current_permission_context
  -> select_configured_backend
       -> keyword_retrieval
       -> pgvector_retrieval
  -> canonical_permission_guard
  -> rank_and_bound_evidence
  -> assemble_server_evidence_slots
  -> generate_structured_answer_blocks
  -> validate_claim_evidence_refs
  -> revalidate_selected_evidence
  -> project_server_citations
  -> record_cost_and_trace
  -> END
```

첫 RAG V2에서는 backend config에 따른 keyword/pgvector edge가 실제 conditional
routing을 증명한다. `fact`, `relation`, `impact`, `timeline` query classification과
Neo4j route는 `neo4j-graphrag` adapter가 존재할 때 별도 GraphRAG spec에서 추가한다.

RAG graph는 request-scoped stateless graph로 시작하며 durable checkpointer를 요구하지
않는다. 따라서 checkpointer 장애가 `/ask`와 `/search` availability를 막지 않는다.

## 6. State, input/output, runtime context

### 6.1 원칙

- checkpointed state에는 JSON-safe primitive와 opaque DB id/hash만 넣는다.
- SQLAlchemy `Session`, session factory, model, retriever, registry, logger, authenticated
  actor는 state에 넣지 않는다.
- 실행 의존성은 `StateGraph(..., context_schema=...)`와 `Runtime[Context]`로 주입한다.
- resume request는 새 DB session과 현재 actor/permission을 runtime context로 제공한다.
- paused thread가 생성 당시 permission snapshot을 현재 권한으로 신뢰하지 않는다.
- Review V2 initial request는 connector/source 원문이나 자유 형식 질문을 durable graph
  input으로 받지 않는다. 대상은 먼저 canonical PostgreSQL source/version 참조로
  고정한다.
- RAG V2는 checkpointer 없이 request-scoped로 실행한다. 질문과 bounded evidence
  content는 해당 요청의 transient state에만 존재하며 checkpoint와 audit에 남지 않는다.

### 6.2 Review graph input/state/output

별도의 `ReviewGraphInput`, `ReviewGraphState`, `ReviewGraphOutput` TypedDict를 사용하며
필드 추가는 schema version 변경으로 취급한다.

`ReviewGraphInput`은 다음 opaque application id만 받는다.

- `workflow_thread_id: str`

graph의 첫 node는 이 id로 `AgentWorkflowThread`, `AgentWorkflowRequest`,
`AgentWorkflowEvidenceRef`를 읽고 현재 permission으로 canonical source/version을 다시
확인한다. initial invocation과 repair는 같은 DB reference를 사용하므로 raw request를
checkpoint에 복제하지 않는다.

durable `ReviewGraphState`는 다음 값만 가진다.

- `workflow_thread_id: str`
- `graph_version: str`
- `input_hash: str`: versioned canonical input의 keyed HMAC
- `evidence_version_hash: str`: ordered immutable evidence refs의 keyed HMAC
- `review_item_ids: list[int]`
- `review_status_counts: dict[str, int]`
- `phase: str`
- `completed_nodes: list[str]`
- `error_codes: list[str]`

`ReviewGraphOutput`은 다음 최소 projection이다.

- `workflow_thread_id: str`
- `status: str`
- `review_item_count: int`
- `review_status_counts: dict[str, int]`
- `error_codes: list[str]`

ReviewItem ids는 내부 state에는 허용하지만 public response에는 직접 노출하지 않는다.

다음 값은 checkpoint에 넣지 않는다.

- 질문 원문과 objective 원문
- source URL, snippet, message body, relationship path
- LLM prompt/output
- provider exception text
- API key, OAuth token, raw connector payload
- SQLAlchemy/model/client instance

state channel 규칙은 명시적으로 정의한다.

- scalar와 producer-owned id list: replacement
- `completed_nodes`: append-unique reducer
- `error_codes`: 허용된 code만 bounded append-unique reducer
- candidate/evidence 본문: state에 저장하지 않고 canonical DB에서 다시 조회

### 6.3 Canonical workflow request/evidence reference

`AgentWorkflowRequest`는 `workflow_thread_id`와 1:1이며 다음 값만 보존한다.

- `input_schema_version`
- `request_kind=review_source_versions`
- normalized `agent_names`
- `selection_policy_version`
- `input_hash`
- `fingerprint_key_version`

`AgentWorkflowEvidenceRef`는 `(workflow_thread_id, ordinal)`을 key로 다음을 보존한다.

- canonical source table/type과 row id
- immutable external revision, `DocumentVersion.id`, 또는 source content signature
- permission level snapshot
- content fingerprint

원문은 이 두 table에 복제하지 않는다. 실제 node는 reference로 기존 permission-aware
canonical source/document-version table을 읽는다. mutable source가 version table을 갖지
않으면 preflight 시 row id와 content signature를 고정하고, 실행 전 signature가 달라지면
`evidence_changed`로 중단해 새 workflow attempt를 요구한다.

`input_hash`, `evidence_version_hash`, candidate content fingerprint는 plain digest가
아니다. 별도 server secret을 사용하는 HMAC-SHA-256으로 만들고, UTF-8 NFC, key 정렬,
stable list ordering을 적용한 canonical JSON 앞에 schema/policy version을 붙인다. key
rotation을 위해 `fingerprint_key_version`을 함께 저장한다.

### 6.4 RAG graph input/state/output

checkpointer가 없는 `RagGraphInput`은 `question: str`만 받고 actor와 permission은 runtime
context에서 받는다. transient `RagGraphState`는 normalized query, selected backend,
bounded `EvidenceSlot` 목록, structured answer blocks와 cost trace를 가진다.
`EvidenceSlot`은 request-local opaque slot id와 canonical record id/version, 모델에 허용된
content만 포함한다. `RagGraphOutput`은 answer text, 사용한 slot ids, hidden count,
backend/fallback, token/cost summary만 반환하고 API mapper가 canonical citation을 붙인다.

### 6.5 Runtime context

`ReviewRuntimeContext`는 request별 session factory, 현재 actor, permission resolver, agent
registry, idempotent draft service와 lease service를 가진다. 각 node가 짧은 session을 열고
닫으며 model 호출 동안 DB transaction이나 row lock을 유지하지 않는다.
`RagRuntimeContext`는 session factory, permission resolver, retriever registry, answer
model과 tool logger를 가진다. retrieval session은 model call 전에 닫고, 생성 후
revalidation은 새 짧은 session에서 수행한다.

runtime context는 checkpoint serialization 대상이 아니다. reducer와 serialization
test는 strict serializer와 PostgreSQL integration 환경에서도 실행한다.

현재 FastAPI와 SQLAlchemy route가 sync이므로 Deliverable C는 sync `PostgresSaver`와
`graph.invoke()`를 선택한다. event loop 안에서 sync invoke를 실행하지 않는다. async
route/`AsyncPostgresSaver` 전환은 별도 migration으로 다룬다.

## 7. Checkpointer lifecycle과 운영 계약

### 7.1 환경별 mode

- test: 테스트마다 새 `InMemorySaver` 주입
- SQLite smoke: application lifespan 동안 하나의 process-local `InMemorySaver`
- production PostgreSQL: application lifespan의 psycopg connection pool을 사용하는
  `PostgresSaver`

SQLite smoke는 single-process에서만 resume을 보장하고 재시작 복구를 보장하지 않는다.
status API에 `durable=false`를 표시한다.

### 7.2 PostgreSQL 연결

`checkpointing.py`는 `Settings.resolved_database_url()`의 SQLAlchemy
`postgresql+psycopg://` URL을 credential을 로그에 남기지 않고 psycopg
`postgresql://` DSN으로 변환한다. pool connection은 PostgresSaver 요구사항에 맞게
`autocommit=True`, `row_factory=dict_row`로 구성한다.

### 7.3 Bootstrap과 readiness

- `PostgresSaver.setup()`은 application import/startup에서 실행하지 않는다.
- Alembic 이후 실행하는 전용 bootstrap command가 한 번 호출한다.
- bootstrap은 실행 전 database backup/restore point를 확인하고 설치된 LangGraph,
  checkpointer package version, checkpoint schema revision, 실행 시각을
  `AgentRuntimeSchemaVersion`에 기록한다.
- readiness는 checkpoint table과 connection을 확인할 뿐 `setup()`을 호출하지 않는다.
- production에서 table/pool이 준비되지 않으면 Review V2 endpoint만 503
  `checkpoint_unavailable`로 fail closed한다.
- in-memory saver로 자동 강등하지 않는다.

initial/resume invoke는 모두 `durability="sync"`를 사용한다. invoke가 반환된 뒤 saver의
`get_tuple(config)`로 동일 thread의 checkpoint id가 저장·진행됐는지 확인한다. pause이면
반환 mapping의 `__interrupt__`와 `graph.get_state(config)`의 pending interrupt도 일치시킨
다음에만 application thread를 `awaiting_human_review`로 바꾼다. terminal run도 saver tuple
진행을 확인한 뒤 status를 바꾼다. 확인이 실패하면 성공 응답을 만들지 않고
`checkpoint_failed` reconciliation으로 보낸다.

### 7.4 Serializer와 retention

- production은 `LANGGRAPH_STRICT_MSGPACK=true`를 필수로 사용한다.
- state는 built-in JSON-safe type만 사용해 custom module deserialization을 요구하지
  않는다.
- checkpoint DB account는 필요한 schema/table에만 최소 권한을 가진다.
- waiting thread는 명시적 cancel/resolve 전 삭제하지 않는다.
- completed/failed checkpoint retention 기본값은 30일이며 configurable prune job이
  처리한다.
- AuditLog와 PostgreSQL official knowledge retention은 checkpoint prune과 분리한다.

## 8. Application workflow persistence model

LangGraph checkpoint metadata만으로 ownership과 product status를 판단하지 않는다.
application-owned `AgentWorkflowThread`를 Alembic migration으로 추가한다.

필수 field:

- `thread_id`: server-issued opaque UUID, unique, 255자 미만
- `workflow_name`
- `graph_version`
- `checkpoint_thread_id`
- `checkpoint_store`
- `owner_subject_id: str`
- `security_scope_id: str`: 현재 single-workspace 배포는 `default`, 향후 tenant/workspace id
- `client_request_id: str | None`
- `input_hash`
- `evidence_version_hash`
- `status`
- `state_version`: optimistic compare-and-swap용 integer
- `lease_token`, `lease_expires_at`: model work claim/recovery용 nullable field
- `checkpoint_confirmed_at`
- `cancelled_at`, `cancelled_by_subject_id`
- `created_at`, `updated_at`, `completed_at`, `expires_at`

초기 `/run`은 client가 `thread_id`를 지정하게 하지 않는다. network retry가 필요한
client는 별도 `client_request_id`를 보내고, DB는
`(security_scope_id, workflow_name, owner_subject_id, client_request_id)`를 partial
unique 처리한다.

같은 unique key가 이미 있으면 canonical `input_hash`, `evidence_version_hash`,
`graph_version`, normalized agent set과 selection policy를 비교한다. 모두 같을 때만 기존
thread와 현재 status를 반환한다. 하나라도 다르면 기존 run을 재사용하지 않고 409
`idempotency_key_reused`를 반환한다.

ReviewItem에는 nullable `workflow_thread_id`, `candidate_key`,
`predecessor_review_item_id`를 추가한다. AgentRun에는 nullable
`workflow_thread_id`, `effect_key`를 추가한다.

- ReviewItem unique: `(workflow_thread_id, candidate_key)`
- AgentRun unique: `(workflow_thread_id, effect_key)`

`candidate_key`와 `effect_key`는 agent/prompt version, immutable source/version ids,
keyed content HMAC, item type을 포함해 deterministic하게 만든다. 기존 전역 `cache_key`는
비용 재사용 신호로 유지하되 write idempotency 근거로 사용하지 않는다.

promotion provenance를 위해 `DecisionRecord`, `HistoryEvent`, `TimelineEvent`, `Todo`에
nullable `source_review_item_id`를 추가하고 non-null 값에 partial unique constraint를
둔다. 기존 row는 null을 허용하지만 V2를 포함한 모든 신규 promotion은 이 값을 반드시
기록한다. decision/history/todo 승인에서 함께 생성되는 `TimelineEvent`도 예외가 아니다.

thread status:

- `created`
- `drafting`
- `checkpoint_pending`
- `awaiting_human_review`
- `resuming`
- `completed`
- `needs_more_evidence`
- `checkpoint_failed`
- `failed`
- `cancelled`

모든 run/resume 상태 변경은 `state_version` compare-and-swap과 row lock으로 동시 실행을
막는다. 다른 process가 이미 resume 중이면 409를 반환한다.

## 9. Transaction, idempotency, failure reconciliation

PostgreSQL business transaction과 LangGraph saver write를 하나의 분산 transaction으로
가장하지 않는다. 다음 saga와 repair 경계를 사용한다.

1. `/runs` preflight가 server UUID thread와 canonical request/evidence refs를 하나의 짧은
   transaction으로 만들거나, 정확히 같은 `client_request_id` request를 재사용한다.
2. draft node는 짧은 claim transaction에서 thread row를 lock하고 `state_version`을
   compare-and-swap해 `drafting` lease를 얻은 뒤 즉시 commit한다.
3. immutable evidence version을 읽고 model/deterministic extraction을 실행하는 동안 DB
   transaction과 row lock을 잡지 않는다.
4. 짧은 persistence transaction이 thread row를 다시 lock하고 lease token,
   `state_version`, evidence signature와 현재 permission을 재검증한다. cancellation이나
   evidence 변경이 있으면 계산 결과를 버리고 write하지 않는다.
5. 같은 transaction에서 AgentRun과 ReviewItem을 thread-bound unique key로
   insert-or-return-existing하고 thread를 `checkpoint_pending`으로 바꾼다. source-agent
   helper는 이 graph-owned session에서 `flush()`만 하고 내부 `commit()`을 하지 않는다.
6. side effect가 없는 `await_human_review` node가 `interrupt()`를 호출하고 sync
   durability로 checkpoint를 저장한다.
7. actual interrupt가 반환되면 route가 saver tuple을 확인하고 별도 짧은 transaction으로
   thread를 `awaiting_human_review`로 전환한다.
8. saver 실패 시 bound ReviewItem을 삭제하지 않고 thread를 `checkpoint_failed`로
   기록하며 sanitized audit event를 남긴다.

repair/retry는 같은 application thread와 idempotency keys를 재사용한다.

- recoverable interrupt checkpoint가 있으면 thread 상태만 reconcile한다.
- checkpoint가 없거나 손상되었으면 새 `checkpoint_thread_id` attempt를 발급하고
  이미 존재하는 ReviewItem/AgentRun을 재사용한다.
- partial saver state를 approval source로 사용하지 않는다.
- duplicate invocation, concurrent unique collision, draft commit 직후 crash,
  model 실행 중 lease 만료/cancel, checkpoint 성공 후 app status update 실패를 각각
  테스트한다.

## 10. Review Queue state machine과 신뢰 경계

현재 Review Queue 상태 전이를 명시적으로 제한한다.

| 현재 상태 | 허용 전이 | promotion |
|---|---|---|
| `pending_review` | `approved`, `rejected`, `needs_more_evidence` | `approved` 전이에서만 1회 |
| `needs_more_evidence` | 새 evidence candidate/thread로 후속 처리 | 없음 |
| `approved` | 상태 전이 없음; 같은 approve 재요청은 stable result 재조회 | 이미 완료 |
| `rejected` | 없음 | 없음 |

`needs_more_evidence`는 trusted knowledge가 아니며 promotion과 RAG context에 포함되지
않는다. 이번 V2에서 해당 상태는 현재 candidate attempt의 종료 결과다. resume 시
graph는 `finalize_needs_more_evidence`로 끝나며 downstream answer나 knowledge write를
실행하지 않는다. 추가 근거가 수집되면 새 candidate version과 새 thread를 만들고 이전
ReviewItem id를 provenance로 연결한다.

resume resolution 규칙:

- 하나라도 `pending_review`이면 409 `review_unresolved`
- 하나라도 `needs_more_evidence`이면 thread outcome은 `needs_more_evidence`
- 나머지가 모두 `approved`/`rejected`이면 `completed`
- 승인 여부는 checkpoint/resume payload가 아니라 현재 PostgreSQL row로만 판단
- approve route는 ReviewItem을 `SELECT ... FOR UPDATE`로 잠그거나 동등한 atomic
  conditional update를 사용
- knowledge insert와 ReviewItem 상태 변경은 하나의 DB transaction에서 처리
- 각 target insert는 `source_review_item_id` unique constraint로
  insert-or-return-existing하고 생성된 canonical ids를 반환
- single approve, `/bulk`, `/approve-agent-candidates`를 포함한 모든 approval entry point는
  같은 helper를 사용한다. multi-item route는 ReviewItem id 오름차순으로 lock해 deadlock을
  피하고 각 item에 동일한 transition/promotion 규칙을 적용한다.
- commit 응답 유실 후 같은 approve를 재요청하면 새 전이가 아니라 기존 canonical
  promotion ids와 `replayed=true`를 반환; 다른 terminal 상태 변경은 409

따라서 두 process가 같은 pending item을 동시에 읽어도 한 transaction만 transition과
insert를 수행한다. 두 번째 transaction은 lock 뒤 `approved`를 관찰하고 기존
`source_review_item_id` row를 조회할 뿐 새 knowledge나 자동 생성 Timeline을 만들지
않는다.

이 state machine은 기존 Review Queue trust boundary에 대한 의도적 contract 변경이다.
implementation plan 작성 전에 사용자 승인을 다시 받아야 한다.

## 11. V2 API와 actor authorization

기존 endpoint는 flag-off와 migration 기간에 그대로 유지한다.

### Legacy 유지

- `GET /api/v1/orchestration/company-memory`
- `POST /api/v1/orchestration/company-memory/dry-run`
- `POST /api/v1/orchestration/company-memory/run`

legacy route와 response field shape는 유지한다. 다만 status는 metadata-only checkpoint를
실제 checkpoint로 표현하지 않는다.
`hitl_checkpointing=false`, `checkpoint_store=none`으로 정정하고 별도
`review_boundary=metadata_only`를 표시한다.

### 신규 V2

- `GET /api/v1/orchestration/v2/company-memory`
- `POST /api/v1/orchestration/v2/company-memory/dry-run`
- `POST /api/v1/orchestration/v2/company-memory/runs`
- `GET /api/v1/orchestration/v2/company-memory/runs/{thread_id}`
- `POST /api/v1/orchestration/v2/company-memory/runs/{thread_id}/resume`
- `POST /api/v1/orchestration/v2/company-memory/runs/{thread_id}/cancel`

V2 `/runs` request는 다음 값만 허용한다.

- `source_refs: list[{source_type, source_id, version_or_signature}]`
- allowlist 검증된 `agent_names: list[str]`
- optional `client_request_id: str`

질문/objective 원문과 임의 prompt는 받지 않는다. route가 source refs를 현재 actor의
permission으로 확인하고 정렬·정규화한 뒤 `AgentWorkflowRequest`와
`AgentWorkflowEvidenceRef`를 먼저 저장한다. graph에는 resulting
`workflow_thread_id`만 전달한다. `/dry-run`도 같은 validation/fingerprint code를 쓰되
thread, reference, checkpoint, ReviewItem을 저장하지 않는다.

initial run은 후보가 있으면 HTTP 202와 actual interrupt에서 매핑한 다음 값을 반환한다.

- `thread_id`
- `status=awaiting_human_review`
- `review_item_count`
- `durable`
- `graph_version`
- `resume_allowed=false`

원문과 숨겨진 item id는 반환하지 않는다. 자세한 ReviewItem은 기존 permission-aware
Review Queue API에서 조회한다.

모든 LangGraph invoke는 다음 config의 같은 `checkpoint_thread_id`를 사용한다.
top-level graph에서 `checkpoint_ns`는 생략해 root namespace(`""`)를 유지한다.

```python
{
    "configurable": {
        "thread_id": workflow_thread.checkpoint_thread_id,
    }
}
```

`checkpoint_ns`는 LangGraph root/subgraph가 관리하는 값이지 application graph version
partition key가 아니다. graph version은 immutable `AgentWorkflowThread.graph_version`과
builder registry로 선택한다. 물리적 thread id 충돌을 추가로 피해야 하면 server가
`{graph_version}:{uuid}` 형식의 opaque `checkpoint_thread_id`를 발급한다.

initial/resume 실행은 sync endpoint에서
`graph.invoke(..., context=runtime_context, durability="sync")`를 사용한다. actual pause
여부는 반환 mapping의 `__interrupt__`와 saver `get_tuple(config)`를 함께 확인한다. API가
임의로 `awaiting_human_review`를 추론하지 않는다.

resume은 DB 상태 확인 후 다음과 같이 acknowledgement만 전달한다.

```python
Command(resume={"event": "review_resolution_checked", "state_version": state_version})
```

authorization:

- foreign thread는 존재 여부를 노출하지 않는 404
- owner는 자신의 thread 상태 조회와 resume 가능
- admin/reviewer는 모든 bound ReviewItem의 현재 permission을 검토할 권한이 있을 때
  resume 가능
- resume 직전에 actor의 현재 allowed permission levels와 각 ReviewItem을 다시 확인
- ReviewItem action 자체가 graph state에 승인 값을 주입하지 않음
- 자동 resume hook은 이번 범위에서 제외

cancellation:

- owner 또는 허용된 admin만 자신의 가시 범위 안 thread를 cancel할 수 있고 foreign
  thread는 404를 유지한다.
- `created`, `drafting`, `checkpoint_pending`, `checkpoint_failed`,
  `awaiting_human_review`에서만 CAS로 `cancelled` 전이가 가능하다.
- `resuming`과 경쟁하면 한 전이만 성공하고 다른 요청은 409를 받는다.
- in-flight worker는 persistence 직전 lease와 thread status를 다시 확인해 cancelled
  결과를 버린다.
- cancel은 bound ReviewItem을 삭제·승인·거절하거나 이미 승인된 knowledge를 되돌리지
  않는다. pending candidate는 Review Queue에서 독립적으로 검토할 수 있지만 cancelled
  graph는 다시 resume되지 않는다.

## 12. Feature flag와 graph version compatibility

독립 flag를 사용한다.

- `LANGGRAPH_REVIEW_V2_ENABLED=false` 기본
- `LANGGRAPH_RAG_V2_ENABLED=false` 기본

Review mode matrix:

| mode | 동작 |
|---|---|
| flag off | legacy endpoint/shape 유지, actual checkpoint 표시 안 함 |
| flag on + SQLite/demo | V2 in-memory, `durable=false`, single-process |
| flag on + ready PostgreSQL | V2 PostgresSaver, `durable=true` |
| flag on + unready PostgreSQL | V2 503, memory fallback 금지 |

`AgentWorkflowThread.graph_version`은 immutable하다. resume은 version registry에서 같은
graph builder를 선택한다. waiting thread가 참조하는 node를 rename/remove하지 않는다.
incompatible deploy 전에는 active thread를 drain/cancel하거나 기존 graph version을
유지한다.

flag가 꺼져도 이미 waiting인 V2 thread를 legacy workflow로 실행하지 않는다. 지원되는
V2 builder가 활성 상태면 재개하고, 제거되었다면 non-mutating
`runtime_version_unavailable`을 반환한다.

## 13. Retriever port와 RAG migration

기존 `VectorStore`는 indexing/vector-store abstraction으로 유지한다. 별도의 search-only
port를 추가한다.

```text
RetrievalRequest
  - query
  - exact allowed_permission_levels
  - limit

RetrievalResult
  - visible evidence candidates
  - bounded hidden_match_count
  - internal backend name
  - latency/call/fallback trace
```

adapter:

- `KeywordEvidenceRetriever`
- `PgVectorEvidenceRetriever`
- 후속 `Neo4jGraphEvidenceRetriever`

`RAG_RETRIEVAL_BACKEND`은 내부 설정으로 시작한다. 이 phase에서는 public request가
임의 backend나 아직 없는 `neo4j`를 선택할 수 없다.

legacy `RAG_USE_PGVECTOR_SEARCH`는 migration alias로 유지한다.

- `RAG_RETRIEVAL_BACKEND`이 설정되면 우선
- 설정되지 않았고 legacy boolean이 true이면 `pgvector`
- 둘 다 없으면 `keyword`

내부 backend 이름은 `keyword`/`pgvector`를 사용하지만 `/search`의 기존 public
`deterministic_lexical` label과 agent metadata의 `keyword` label은 API mapper에서
보존한다.

cutover 순서:

1. shadow mode에서 기존 서비스와 V2 retriever result ids/hidden count 비교
2. `/ask`
3. `/search`
4. assistant의 모든 `answer_question_with_rag` caller
5. parity 확인 후 legacy retrieval branch 제거

company-memory legacy run은 Review V2와 독립적으로 유지하고 RAG V2에 암묵적으로
연결하지 않는다.

## 14. Permission, cache, hidden-match guardrail

- 현재 actor의 정확한 `allowed_permission_levels`를 사용하고 role 이름에서 다시
  추론하지 않는다.
- unknown permission level은 `internal`로 간주하지 않고 fail closed한다.
- source와 chunk/index metadata가 다르면 strictest permission을 사용한다.
- retriever query에서 permission filter를 적용한다.
- canonical PostgreSQL record를 다시 읽는 guard node에서 한 번 더 확인한다.
- fallback retriever도 동일 PermissionContext와 evidence limit를 사용한다.
- 권한 없는 source id, URL, snippet, relationship path는 model context와 API에 넣지
  않는다.
- cache key에는 evidence hash, prompt/model version, exact permission fingerprint,
  backend와 retrieval policy version을 포함한다.

`hidden_match_count`는 전체 unauthorized corpus 크기가 아니다. 동일 query의 relevance
threshold와 bounded candidate window 안에서 visibility filter로 제외된 수이며 최대
scan limit으로 cap한다. 이 정의를 keyword와 pgvector가 동일하게 구현한다.

## 15. Citation 계약

첫 RAG V2에서는 현재 public response compatibility를 유지한다.

- server는 permission-filtered bounded window에 request-local `E1`, `E2` 같은 opaque
  evidence slot id를 부여한다. 모델에는 허용된 content와 slot id만 보내고 canonical
  row id, URL, permission metadata는 보내지 않는다.
- LangChain structured output은
  `answer_blocks: list[{text: str, evidence_slot_ids: list[str]}]`와
  `insufficient_evidence_reason: str | None`만 허용한다.
- server는 block 순서대로 public answer text를 조립한다. 일반 answer의 모든 substantive
  block은 최소 1개의 존재하는 slot을 참조해야 하며 unknown/unselected slot은 전체
  answer를 `citation_validation_failed`로 fail closed한다.
- 생성 후 선택된 slot의 canonical row/version과 현재 permission을 다시 읽는다.
  revoked, stale, permission-changed evidence가 하나라도 있으면 모델 text를 반환하지
  않는다.
- citation URL, snippet, type, permission은 모델이 아니라 사용된 canonical evidence에서
  투영하고, 선택되지 않은 window 전체를 citation으로 붙이지 않는다.
- evidence가 없으면 `answer_blocks=[]`와 명시적인 insufficient-evidence reason만 허용하고
  기존 API의 안전한 안내문과 빈 citation으로 변환한다.

golden query의 block-to-evidence faithfulness 평가가 legacy baseline보다 낮거나 structured
mapping이 구현되지 않으면 RAG V2는 shadow mode에 머물며 `/ask`, `/search`, assistant를
cutover하지 않는다.

## 16. 오류, fallback, audit, data minimization

error category:

- `invalid_input`
- `idempotency_key_reused`
- `evidence_changed`
- `permission_denied`
- `retriever_not_configured`
- `retriever_unavailable`
- `checkpoint_unavailable`
- `checkpoint_failed`
- `review_unresolved`
- `runtime_version_unavailable`
- `model_unavailable`
- `citation_validation_failed`
- `concurrent_resume`
- `invalid_state_transition`

permission/citation failure는 fallback으로 우회하지 않는다. pgvector runtime 장애만
동일 permission/bound의 keyword fallback을 허용하며 response와 trace에
`fallback_used=true`를 표시한다.

필수 audit event:

- thread created
- draft committed/reused
- checkpoint confirmed/failed/reconciled
- resume attempted/denied/succeeded
- thread resolved/cancelled
- retrieval backend/fallback
- permission/citation failure
- token/cost summary

audit metadata는 actor id, thread id, graph/backend version, outcome/error category와
bounded count만 allowlist 방식으로 기록한다. 질문, source URL/snippet, provider error
text, model output은 checkpoint와 audit에 넣지 않는다.

## 17. 테스트 전략

모든 behavior change는 RED를 먼저 확인한다.

### Deliverable A

- exact dependency resolution
- 모든 provider adapter import
- fake model `create_agent` structured response
- direct model structured output validation
- StateGraph conditional edge와 interrupt import/invoke
- PostgresSaver import

### Deliverable B

- Alembic fresh schema와 existing schema upgrade
- application lifespan saver lifecycle
- SQLAlchemy URL의 safe DSN 변환과 secret 미노출
- bootstrap/readiness 분리
- strict msgpack serialization
- checkpoint tuple에 질문/URL/snippet/model output이 없음을 직접 검사
- saved tuple의 top-level `checkpoint_ns`가 root `""`인지 확인하고 restart 후 같은
  `checkpoint_thread_id`로 resume
- initial/resume의 `durability="sync"`와 saver tuple 확인 전 application status가
  `awaiting_human_review`로 바뀌지 않음을 검증
- graph reducer replay/idempotency
- graph version registry와 flag mode matrix
- SQLite restart 시 non-durable error

### Deliverable C

- 후보 없으면 interrupt 없이 종료
- 후보 있으면 실제 interrupt 후 downstream 미실행
- actual interrupt payload와 HTTP 202 mapping
- 동일 checkpoint thread id와 `Command(resume=...)`
- DB가 resume payload보다 approval authority임
- pending item이 있으면 409
- needs-more-evidence branch에서 promotion/RAG 미실행
- foreign thread 404와 reviewer permission 확인
- concurrent resume 한 건만 성공
- duplicate initial request가 같은 records 반환
- 같은 `client_request_id`와 다른 canonical input/evidence/params는 409
  `idempotency_key_reused`
- draft commit 직후 crash와 checkpoint failure repair
- model work 중 cancel/lease expiry는 persistence 없음
- app restart 후 PostgreSQL checkpoint resume
- single/bulk/concurrent approve가 target table별 한 record만 생성하고, decision/history/todo
  자동 Timeline도 `source_review_item_id`로 한 번만 생성
- approve commit 응답 유실 뒤 retry는 같은 canonical ids와 `replayed=true`를 반환
- approved/rejected/needs-more-evidence terminal 상태 간 불법 전이는 409

### Deliverable D

- conditional keyword/pgvector routing
- legacy config precedence
- public backend label compatibility
- exact permission context 전달과 unknown level fail closed
- bounded hidden-match 동일성
- pgvector 장애의 same-context keyword fallback
- structured answer block의 valid/unknown/uncited evidence slot 검증
- 선택된 slot만 server citation으로 투영하고 revoked/stale evidence는 fail closed
- claim-to-evidence faithfulness golden evaluation이 legacy baseline 이상
- insufficient-evidence contract
- `/ask`, `/search`, assistant shadow parity와 순차 cutover
- fake model만으로 전체 graph 실행

### Regression

- agent runtime contracts
- Review Queue promotion/RBAC
- RAG orchestrator와 assistant persistence
- orchestration API/frontend dry-run/status
- pgvector adapter/integration fixture
- focused backend suite
- 전체 backend baseline 대비 신규 실패 0개
- 관련 frontend Playwright와 production build

implementation plan을 시작할 때 fresh baseline을 다시 기록한다. 기존 실패가 있으면
이번 변경과 무관한지 분류하며, 신규 실패를 기존 문제로 표시하지 않는다.

## 18. Rollout과 rollback

1. dependency-only commit과 compatibility verification
2. runtime/schema/checkpoint primitives, route cutover 없음
3. disabled-by-default Review V2 endpoint
4. PostgreSQL interrupt/restart/resume smoke
5. disabled-by-default RAG V2 shadow mode
6. `/ask`, `/search`, assistant 순차 cutover
7. 안정화 후 legacy linear wrapper와 flags 제거

rollback은 deliverable 단위로 수행한다.

- `PostgresSaver.setup()` 전 dependency 문제: 이전 `pyproject.toml`/`uv.lock` commit으로
  복귀
- `PostgresSaver.setup()` 후 문제: bootstrap 기록과 database backup을 기준으로 먼저
  staging에서 N-1 package가 현재 checkpoint schema를 읽고 waiting thread를 resume할 수
  있는지 확인한다. 호환되지 않으면 package만 downgrade하지 않는다. 신규 V2 run을
  flag로 중단하고 upgraded checkpointer와 기존 graph builders를 유지한 채 active thread를
  drain/cancel한다. checkpoint schema 복구는 승인된 backup restore/change procedure로만
  수행한다.
- Review V2 문제: 신규 run 생성만 flag로 중단하고 active V2 thread는 같은 graph
  version으로 resolve/cancel
- RAG V2 문제: legacy retriever service로 복귀하되 이미 저장된 AgentRun/AuditLog는
  삭제하지 않음
- permission/citation 이상: 해당 V2 path 즉시 fail closed

## 19. 성공 기준

- status API가 metadata-only checkpoint를 실제 durable checkpoint로 과장하지 않는다.
- Review V2가 실제 LangGraph interrupt/resume과 PostgreSQL saver로 동작한다.
- DB session과 sensitive evidence가 checkpoint에 저장되지 않는다.
- retry/concurrency/crash가 ReviewItem, AgentRun, promotion 중복을 만들지 않는다.
- idempotency key payload mismatch가 기존 thread를 잘못 재사용하지 않는다.
- Review Queue와 PostgreSQL promotion이 계속 trust source of truth다.
- RAG V2가 실제 conditional edge와 공통 retriever port를 사용한다.
- permission, hidden-match, claim-to-evidence citation, cache, cost 계약이 기존보다
  약해지지 않는다.
- `/ask`, `/search`, assistant public response compatibility가 유지된다.
- Neo4j retriever가 API route나 source agent를 수정하지 않고 등록될 수 있다.
- 각 deliverable이 독립적인 test evidence와 rollback point를 가진다.

## 20. GraphRAG 후속 순서

1. Neo4j foundation과 최소 권한 connection
2. PostgreSQL approved-knowledge outbox projection
3. idempotent graph node/edge sync와 revoke/supersede
4. `neo4j-graphrag` VectorCypher/HybridCypher adapter
5. structured query classification과 Neo4j conditional branch
6. pgvector baseline 비교 평가
7. 실제 Neo4j-backed Knowledge Map

## 21. 공식 참고

- [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph StateGraph runtime context](https://reference.langchain.com/python/langgraph/graph/state/StateGraph)
- [LangGraph Postgres checkpointer](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-postgres)
- [LangChain agents](https://docs.langchain.com/oss/python/langchain/agents)
- [LangChain structured output](https://docs.langchain.com/oss/python/langchain/structured-output)
