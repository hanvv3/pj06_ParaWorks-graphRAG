# ParaWorks Review Queue HITL V2 설계

- 검토 버전: 1
- 설계 승인일: 2026-08-27
- 상태: 설계 승인, 구현 계획 및 제품 코드 구현 미승인

상위 설계는
`docs/superpowers/specs/2026-08-26-langchain-langgraph-runtime-foundation-design.md`다.
이 문서는 그중 Deliverable C만 구체화한다. CDC/outbox 도입 시점은 이 문서의
사용자 결정을 우선한다.

## 1. 결정 요약

Deliverable C는 기존 설명용 `hitl_checkpoint` dictionary를 실제 LangGraph
human-in-the-loop workflow로 교체한다. 새 V2 workflow는 Review Queue 후보 생성과
검토 완료 조정만 담당하며, RAG 답변 생성이나 GraphRAG projection을 수행하지 않는다.

확정한 원칙은 다음과 같다.

1. LangGraph 1.x의 실제 `interrupt()`와 같은 checkpoint thread의
   `Command(resume=...)`를 사용한다.
2. 사용자는 새 Agent Runs 화면을 거치지 않는다. 주 경로는
   `Integrations -> 검토 후보 만들기 -> Review -> 검토 완료` 두 화면이다.
3. 승인 값은 resume payload가 아니라 현재 PostgreSQL `ReviewItem` 상태에서 읽는다.
4. 모든 Review 상태 전이는 하나의 공통 service를 통하고, 승인 promotion은 재시도와
   동시 실행에서도 정확히 한 번만 발생한다.
5. V2 endpoint는 기본 비활성화하고 기존 V1 endpoint는 migration과 rollback을 위해
   유지한다. 같은 source batch를 V1과 V2가 동시에 처리하지 않는다.
6. Slack, RAG cutover, Neo4j, Knowledge Map 변경, CDC, transactional outbox, message
   broker, stream consumer는 Deliverable C에서 제외한다.
7. CDC/streaming은 대규모 데이터의 실제 병목이 계측된 뒤 별도 설계와 승인을 거쳐
   검토한다. Deliverable D나 GraphRAG의 선행 조건으로 간주하지 않는다.

## 2. 현재 구조와 해결할 문제

현재 저장소에는 Deliverable B에서 완성한 다음 기반이 이미 있다.

- LangChain 1.3.17, LangGraph 1.2.11과 provider/checkpointer 호환성 계약
- checkpoint-safe `ReviewGraphState`와 `ReviewRuntimeContext`
- `AgentWorkflowThread`, `AgentWorkflowRequest`, `AgentWorkflowEvidenceRef`
- `GraphVersionRegistry`, `CheckpointRuntime`, saver confirmation과 retention
- `ReviewItem(workflow_thread_id, candidate_key)` unique key
- `AgentRun(workflow_thread_id, effect_key)` unique key
- knowledge table별 nullable `source_review_item_id`와 partial unique index
- `LANGGRAPH_REVIEW_V2_ENABLED=false` 기본 flag
- process-local memory mode와 durable PostgreSQL mode

하지만 제품 동작에는 다음 간극이 남아 있다.

- 기존 company-memory graph는 conditional edge, checkpointer, 실제 interrupt/resume이
  없는 직선형 wrapper다.
- 기존 `hitl_checkpoint`는 ReviewItem id를 담은 설명용 metadata이며 graph를 멈추지
  않는다. 후보 생성 후 RAG 단계까지 이미 실행된다.
- public V2 run/status/resume/cancel route가 없다.
- workflow foundation model과 checkpoint primitive가 Review Queue 동작에 연결되지
  않았다.
- single approve, bulk approve, approve-agent-candidates가 상태 변경과 promotion을
  각자 구현한다.
- 현재 promotion은 `source_review_item_id`를 쓰지 않아 이미 존재하는 unique index가
  중복 knowledge와 companion Timeline을 막지 못한다.
- Review page에는 workflow별 filter, 진행 상태, 명시적 resume control이 없다.
- Integration sync는 `changed_source_ids`만 반환하며 immutable source version
  reference를 V2에 전달하는 계약이 없다.

Deliverable C는 이 간극만 닫는다. 기존 Review evidence drawer, card, project selection,
bulk selection UX와 approved knowledge 화면은 다시 설계하지 않는다.

## 3. 범위

### 3.1 포함

- `company-memory-review-v2.0` 실제 Review HITL graph
- canonical source/version preflight와 workflow thread 생성
- non-Slack allowlisted agent의 idempotent candidate/AgentRun draft
- 실제 interrupt 저장 확인과 PostgreSQL restart resume
- Review Queue 공통 상태 전이 및 exactly-once promotion service
- V2 run, dry-run, status, resume, cancel API
- integration sync 결과의 additive `changed_source_refs`
- Review API의 workflow filter와 V2 workflow summary projection
- Integrations와 Review 두 화면의 최소 end-to-end UI
- checkpoint failure reconciliation, cancellation, permission recheck, audit
- fake/deterministic model 및 PostgreSQL integration test

### 3.2 제외

- Slack source, Slack agent, Slack fixture와 현재 보류된 Slack 실패 수정
- `/ask`, `/search`, assistant, RAG answer graph와 pgvector routing 변경
- Neo4j, `neo4j-graphrag`, Knowledge Map 변경
- 새로운 candidate type이나 knowledge schema
- 자동 resume hook
- 새 Agent Runs primary navigation 또는 workflow 관리 전용 화면
- live connector, OAuth provider, live LLM, live embedding을 호출하는 자동 테스트
- CDC connector, database log tailing, transactional outbox table, broker topic,
  stream consumer, projection worker
- V1 route 제거

### 3.3 CDC/streaming 유예 계약

Deliverable C의 data plane은 기존 connector sync와 명시적 API/job 실행을 유지한다.
새 outbox schema나 event infrastructure를 미리 만들지 않는다. 미래 전환을 방해하지
않도록 다음 기존 경계만 안정적으로 유지한다.

- connector는 공통 `SourceEvent`를 반환한다.
- sync는 canonical `changed_source_refs` batch를 만든다.
- workflow draft와 Review transition은 trigger transport와 분리된 service다.
- 모든 side effect는 thread/candidate/effect/provenance key로 idempotent하다.

CDC/streaming 재검토는 다음 현상 중 하나 이상이 운영 지표로 확인될 때 별도 spec으로
진행한다.

- ingestion backlog가 지속적으로 증가한다.
- 합의한 freshness SLO를 현재 job 방식으로 지키지 못한다.
- 같은 source change를 여러 독립 projection consumer가 처리해야 한다.
- polling이나 batch worker가 DB 또는 worker 병목의 주원인으로 측정된다.

측정 전에는 임의 처리량 숫자를 도입 기준으로 정하지 않는다. 도입 시에도 Review
state machine과 domain service는 유지하고 trigger transport만 교체하는 것을 목표로
한다.

## 4. 사용자 경험

### 4.1 UX 원칙

- 사용자는 connector와 Review Queue라는 이미 익숙한 두 장소만 사용한다.
- workflow/thread/checkpoint는 사용자에게 기술 용어로 노출하지 않는다.
- 후보 생성은 비용과 범위를 확인할 수 있는 명시적 사용자 action이다.
- review 완료 후 graph 재개도 명시적 action이다. 승인 버튼이 자동으로 graph를
  resume하지 않는다.
- recoverable checkpoint 오류는 사용자의 검토 결과를 지우지 않고 같은 화면에서
  재시도할 수 있어야 한다. terminal `failed`/`cancelled`은 아래 고정 안내를 따른다.

### 4.2 정상 흐름

```text
Integrations
  -> Gmail/Drive/Calendar 동기화
  -> 변경된 자료 수와 예상 비용 표시
  -> "검토 후보 만들기" 클릭
  -> 후보 생성 중
  -> 실제 interrupt가 확인되면 Review로 이동

Review?workflow_thread_id=<opaque id>
  -> 해당 workflow 후보만 우선 표시
  -> 기존 evidence drawer와 승인/거절/추가 근거 요청 사용
  -> 모든 항목이 terminal이면 "검토 완료" 활성화
  -> 클릭하면 같은 thread를 명시적으로 resume
  -> "검토 완료" 또는 "추가 근거 필요" 결과 표시
```

Integrations의 `검토 후보 만들기` 버튼 자체가 paid run의 명시적 확인이다. 버튼 옆이나
바로 위에 dry-run으로 얻은 source 수, 예상 token/cost, budget 상태를 먼저 표시한다.
추가 wizard나 확인 페이지를 만들지 않는다. 비용이 발생하지 않는 deterministic/cache
경로는 `추가 비용 없음`으로 표시한다.

후보가 없으면 Review로 이동하지 않고 Integrations에서 `새 검토 후보 없음`을 표시한다.
후보가 있으면 actual interrupt와 saver tuple 확인이 끝난 뒤에만 Review link를
활성화한다.

### 4.3 Review 화면

기존 `frontend/src/app/review/page.tsx`를 유지하고 다음 context panel만 추가한다.

- workflow 제목과 source connector label
- `검토 완료 항목 / 전체 항목` 진행률
- 현재 상태의 한국어 label
- resolution-ready `awaiting_human_review`에서는 `검토 완료`, recoverable
  `checkpoint_failed`에서만 `다시 시도` primary action
- checkpoint mode가 memory이면 `이 실행은 앱 재시작 후 이어갈 수 없습니다` 안내

workflow filter는 기존 grouping, pagination, bulk action과 함께 동작한다. filter가
없으면 기존 전체 pending queue가 그대로 보인다. sidebar에 Agent Runs나 workflow
항목을 새로 만들지 않는다.

사용자 표시 상태는 다음과 같다.

| 내부 상태 | 사용자 표시 |
|---|---|
| `created`, `drafting`, `checkpoint_pending` | 검토 준비 중 |
| `awaiting_human_review` | 검토 필요 |
| `resuming` | 검토 결과 반영 중 |
| `completed` | 검토 완료 |
| `needs_more_evidence` | 추가 근거 필요 |
| `checkpoint_failed` | 재시도 필요 |
| `failed` | 처리 실패 |
| `cancelled` | 취소됨 |

`failed`와 `cancelled`은 terminal이며 `다시 시도` action을 노출하지 않는다. 같은 exact
batch를 새 thread로 우회 실행하지도 않는다. UI는 Integrations status card와, bound
item이 있는 경우 Review context panel에 `데이터 변경 후 다시 동기화` 안내와 해당
connector의 Integrations deep link를 표시한다. source version/signature, versioned
selection policy, 또는 graph version이 달라진 다음 eligible V2 batch만 새 실행이 된다.
이 제한은 별도 retry-alias schema 없이 stable shared ownership과 audit 의미를 보존하기
위한 의도된 계약이다.

## 5. Architecture와 책임 경계

```text
Integrations UI
  -> connector sync API
  -> changed_source_refs
  -> Review Workflow V2 API
  -> ReviewWorkflowService
       -> canonical evidence resolver
       -> AgentRegistry / draft adapters
       -> LangGraph company-memory-review-v2.0
       -> CheckpointRuntime
  -> actual interrupt
  -> Review UI
       -> existing Review API
       -> shared ReviewTransitionService
       -> PostgreSQL knowledge promotion
       -> explicit workflow resume
```

### 5.1 API route

Route는 authentication, schema validation, HTTP mapping만 담당한다. LangChain이나
LangGraph를 직접 호출하지 않는다. 모든 graph invocation은
`backend/app/agent_runtime/` service 뒤에 둔다.

### 5.2 ReviewWorkflowService

다음 orchestration 책임을 가진다.

- feature flag와 checkpointer readiness preflight
- canonical source/version resolution과 permission 확인
- request/evidence HMAC과 client idempotency 판정
- application thread/lease lifecycle
- graph builder 선택과 initial/resume invocation
- returned interrupt와 saved tuple confirmation
- application status reconciliation, cancel, bounded audit

### 5.3 V2 graph

Graph node는 작고 독립적으로 테스트 가능해야 한다. node는 route나 frontend shape를
모르며, DB/model/client/session을 durable state에 넣지 않는다.

### 5.4 Draft service와 AgentRegistry

V2는 agent module을 직접 import해 서로 호출하지 않는다. `AgentRegistry`의 manifest와
allowlist를 사용한다. C에 필요한 non-Slack adapter는 candidate persistence envelope를
반환하고 내부 `commit()`을 하지 않도록 좁게 분리한다. graph-owned transaction이
AgentRun과 ReviewItem을 함께 insert-or-return-existing한다.

### 5.5 ReviewTransitionService

single approve, bulk approve/reject, approve-agent-candidates, reject,
needs-more-evidence가 공통 상태 전이 규칙을 사용한다. promotion은 이 service 안의 같은
transaction에서 수행한다. 이 경계는 V2 item뿐 아니라 기존 Review API entry point에도
적용해 trust boundary를 하나로 만든다.

### 5.6 Frontend

- Integrations는 sync, dry-run cost, V2 launch와 Review deep link만 담당한다.
- Review는 evidence inspection, 상태 변경, workflow progress와 explicit resume만
  담당한다.
- frontend는 approval을 resume payload에 복제하지 않는다.

## 6. Data 계약

### 6.1 Canonical source reference

공개 request의 source reference는 다음 세 field만 가진다.

```json
{
  "source_type": "gmail",
  "source_id": "opaque-external-or-canonical-id",
  "version_or_signature": "immutable-version-or-content-signature"
}
```

source refs는 `(source_type, source_id, version_or_signature)` 순으로 정렬·dedupe한다.
server는 현재 actor가 볼 수 있는 canonical PostgreSQL row와 version/signature를 다시
확인한다. mutable source가 바뀌었으면 `409 evidence_changed`로 끝내고 새 sync/dry-run을
요구한다.

`ConnectorSyncResult`와 authenticated integration sync response에는 기존
`changed_source_ids`를 유지하면서 additive `changed_source_refs`를 추가한다. 이 변경은
새 batch table을 요구하지 않는다. actor에게 허용되지 않은 source ref는 응답과 V2
request에 포함하지 않는다.

integration candidate-generation mode는 feature flag로 하나만 선택한다.

- V2 flag off: 현재 legacy inline agent/project-assignment bridge를 유지한다.
- V2 flag on: sync는 canonical source 저장과 `changed_source_refs` 반환까지만 수행하고,
  같은 refs에 legacy inline candidate generation을 호출하지 않는다. 모든 candidate
  generation은 사용자의 명시적 V2 launch 안에서 수행한다.

changed source가 없는 recovery path도 V2 mode에서는 자동 candidate를 만들지 않는다.
따라서 동일 source batch가 legacy와 V2 ReviewItem을 중복 생성하지 않는다.

비동기 sync에서도 refs를 잃지 않도록 ingestion commit 뒤 변경된 각 `Source`의
`raw_metadata.last_changed_sync_job_id`를 현재 `SyncJob.job_id`로 기록한다. runtime
status는 최신 job id와 일치하는 canonical Source만 다시 읽어 `changed_source_refs`를
투영한다. 별도 batch/outbox table이나 AuditLog의 source-id 목록에 의존하지 않는다.
후속 sync가 같은 Source를 다시 바꾸면 marker는 최신 job으로 교체되며 UI도 최신
완료 job만 launch 대상으로 삼는다.

Integrations는 최신 sync의 `job_id`와 `selection_policy_version`으로 길이가 제한된
stable `client_request_id`를 만든다. 같은 완료 job에서 double click, polling response
loss, page-level retry가 발생해도 같은 id와 canonical batch로 기존 thread를 재조회한다.
이 값은 canonical batch identity가 아니라 thread를 실제 생성한 actor에게만 귀속되는
optional transport-retry key다. 다른 actor의 existing shared thread를 재사용한 요청에는
그 actor와 key의 별도 alias row를 만들거나 key reservation을 약속하지 않는다.

2026-08-27 사용자 결정에 따라 exact source-version batch ownership은 개인이 아니라
회사/workspace `security_scope_id`에 속한다. batch identity는 다음 값의 keyed HMAC이다.

- `security_scope_id`
- `workflow_name`과 immutable `graph_version`
- `evidence_version_hash`
- normalized exact `agent_names`
- `selection_policy_version`

`owner_subject_id`는 최초 실행자와 audit/cancel 책임을 보존하지만 ownership key에는
들어가지 않는다. 같은 scope에서 matching V2 thread가 있으면 현재 actor가 모든 bound
evidence를 볼 수 있는 경우 기존 thread와 status를 반환하고, 볼 수 없으면 존재 여부를
숨기는 404를 반환한다. 다른 owner/client request id로 parallel thread를 만들지 않는다.

PostgreSQL preflight는 batch HMAC에서 유도한 transaction-scoped advisory lock을 얻은 뒤
matching thread 조회와 생성을 수행한다. hash collision은 불필요한 직렬화만 만들고
identity 판정은 전체 stored hashes를 다시 비교한다. SQLite/demo는 single-process
runtime lock을 사용하며 durable multi-process 보장을 주장하지 않는다. 새 batch unique
column이나 outbox table은 추가하지 않는다. advisory key는 transient이며 실제 match는
기존 `AgentWorkflowThread`의 scope/workflow/graph/evidence fields와 1:1
`AgentWorkflowRequest`의 agent names/selection policy fields를 함께 비교한다.

matching thread는 status나 effect count와 관계없이 그대로 재사용한다. `failed` 또는
`cancelled`인 zero-effect thread도 stable `client_request_id`와 existing unique index를
보존하며 새 thread로 교체하지 않는다. 둘은 user retry 대상이 아니며
`retry_allowed=false`다. transient checkpoint failure만 같은 thread의 repair 경계를
사용한다. exact batch를 다시 처리하려면 source version/signature, versioned
`selection_policy_version`, 또는 `graph_version`이 달라져야 한다.

matching V2 thread가 있으면 legacy inline path도 해당 batch를 처리하지 않는다.
flag-off rollback 중 V1 fallback은 matching V2 thread가 없는 batch에만 허용한다. 이
검사는 backend service에서 강제하며 frontend 조건에 의존하지 않는다. Slack sync는
이 mode 전환과 ownership 검사의 대상이 아니다.

이 계약은 exact canonical batch만 중복 제거한다. 다른 source set/version의 의미상
유사 후보를 fuzzy merge하거나 자동 승인하지 않는다. 그런 후보는 각각 Review Queue에
남고 별도 duplicate-resolution 설계 없이는 trusted knowledge를 자동 병합하지 않는다.

V1에서 이미 처리한 historical batch가 V2 cutover 뒤 다시 후보를 만들지 않도록 source
metadata에 review-generation waterline을 둔다.

- V2-mode sync commit: `review_batch_mode=v2_explicit`,
  `review_batch_signature=<current content_signature>` 기록
- legacy inline candidate 성공: 같은 current signature에
  `review_batch_mode=legacy_inline` 기록
- V2 preflight: 모든 ref의 current signature가 marker와 같고 mode가
  `v2_explicit`일 때만 신규 shared thread 생성 허용

legacy marker이거나 marker 이전 historical source는 V2 신규 batch 대상이 아니며 기존
allowlisted `evidence_changed` 409와 새 V2-mode sync 안내를 반환한다. rollback 중 V1이
V2-eligible batch를 처리하면 성공 transaction에서 marker를 `legacy_inline`으로 바꿔
후속 V2 중복 실행을 막는다. matching V2 thread가 이미 있으면 V1은 먼저 억제되므로
그 thread의 marker는 바꾸지 않는다. marker는 기존 `Source.raw_metadata`를 사용하며
schema migration을 요구하지 않는다.

현재 Deliverable C에서 canonical mapping은 다음과 같이 고정한다.

- `source_type`: `gmail`, `gmail_attachment`, `drive`, `calendar` 중 하나
- `source_id`: integer PK나 provider의 bare id가 아니라 prefix가 포함된
  `Source.source_id`; request의 `source_type`과 DB row가 반드시 일치
- `version_or_signature`: ingestion commit 뒤 canonical Source의
  `raw_metadata.content_signature`; parsed document가 있으면 current
  `DocumentVersion`과 parser revision/signature도 server가 함께 재검증

같은 provider bare id가 있어도 prefix와 source type 경계를 넘어 해석하지 않는다.
형식이 잘못된 ref는 400, 존재하지 않거나 actor에게 보이지 않는 ref는 존재 여부를
숨기는 404, signature mismatch는 409 `evidence_changed`다. refs는 connector event가
아니라 ingestion commit 후 canonical DB row에서 만든다.

### 6.2 Application persistence

Deliverable B의 다음 schema를 그대로 사용한다.

- `AgentWorkflowThread`
- `AgentWorkflowRequest`
- `AgentWorkflowEvidenceRef`
- `ReviewItem.workflow_thread_id`, `candidate_key`, `predecessor_review_item_id`
- `AgentRun.workflow_thread_id`, `effect_key`
- knowledge table의 `source_review_item_id`

새 Alembic migration은 예상하지 않는다. 구현 중 새 public/schema field가 필요하다고
확인되면 output schema/Review trust-boundary human gate에서 멈추고 다시 승인받는다.

### 6.3 Durable graph state

Graph input은 `workflow_thread_id` 하나만 받는다. durable state는 Deliverable B의
`ReviewGraphState`를 사용한다.

- workflow/thread id와 immutable graph version
- keyed input/evidence HMAC
- 내부 ReviewItem ids
- bounded status counts, phase, completed node, allowlisted error code

checkpoint에는 source URL/snippet/body, question/objective, prompt/model output, provider
exception, token, OAuth/API key, SQLAlchemy session/client를 넣지 않는다. public V2
response에는 내부 ReviewItem id를 노출하지 않는다. 상세 evidence는 기존
permission-aware Review API만 반환한다.

### 6.4 Review state machine

| 현재 상태 | 허용 전이 | promotion |
|---|---|---|
| `pending_review` | `approved`, `rejected`, `needs_more_evidence` | `approved`에서만 1회 |
| `approved` | 같은 approve 결과 replay만 허용 | 기존 결과 재조회 |
| `rejected` | 없음 | 없음 |
| `needs_more_evidence` | 없음; 새 candidate/thread 필요 | 없음 |

terminal 상태를 다른 terminal 상태로 바꾸는 요청은 `409 invalid_state_transition`이다.
`needs_more_evidence`는 trusted knowledge와 RAG context에 들어가지 않는다. 후속 sync가
새 candidate를 만들면 `predecessor_review_item_id`로 이전 item을 연결한다.

### 6.5 Exactly-once promotion

승인 transaction은 ReviewItem을 id 오름차순으로 lock하고 다음을 수행한다.

1. 현재 permission과 evidence 요구사항 재검증
2. `pending_review -> approved` 상태 전이
3. target knowledge와 companion Timeline에 모두 `source_review_item_id` 기록
4. partial unique index로 insert-or-return-existing
5. canonical record ids와 timeline ids 반환
6. AuditLog 기록 후 commit

commit 응답 유실 뒤 같은 approve를 재요청하면 기존 canonical ids와
`replayed=true`를 반환한다. concurrent single/bulk approve에서도 한 transaction만
insert하고 나머지는 같은 결과를 읽는다. cache key는 write idempotency로 사용하지
않는다.

### 6.6 Workflow progress projection

`AgentWorkflowThread.status`는 workflow lifecycle만 나타낸다. Review action이 이를
자동으로 `completed`로 바꾸지 않는다. `GET .../runs/{thread_id}`는 호출할 때마다
현재 permission을 확인한 뒤 bound ReviewItem을 PostgreSQL에서 집계해
`review_item_count`와 `review_status_counts`를 만든다. checkpoint에 저장된 이전 count를
public progress의 source of truth로 사용하지 않는다.

status projection은 `review_resolution_ready`, `checkpoint_resumable`,
`resume_allowed`를 분리한다.

- `review_resolution_ready=true`: actor가 모든 bound item을 볼 수 있고, item이 하나
  이상이며, `pending_review`가 0
- `checkpoint_resumable=true`: 현재 checkpoint runtime이 ready이고 같은 thread의
  saver tuple이 실제 resume 가능한 상태로 확인됨
- `resume_allowed=true`: thread가 `awaiting_human_review`이고 위 두 값이 모두 true

memory process restart, PostgreSQL unready, missing/corrupt tuple에서는
`checkpoint_resumable=false`, `resume_allowed=false`, nullable
`resume_error_code=checkpoint_unavailable`을 반환한다. `needs_more_evidence`도 terminal
review 결과이므로 resolution-ready가 되고, 실제 resume 뒤 thread가
`needs_more_evidence`로 끝난다. bulk 일부 실패/skip은 남은 pending count로 즉시
보이고, approval replay는 count를 바꾸지 않는다. cancelled/failed/completed thread는
항상 `resume_allowed=false`다. `retry_allowed=true`는 authorized actor의
`checkpoint_failed` thread에서 checkpoint runtime이 현재 ready일 때만 반환한다.

## 7. LangGraph workflow

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

`await_human_review`는 side effect가 없으며 다음 interrupt만 발생시킨다.

```python
interrupt({
    "event": "review_resolution_required",
    "state_version": state_version,
})
```

resume은 같은 `checkpoint_thread_id`와 root `checkpoint_ns == ""`에서 다음
acknowledgement만 전달한다.

```python
Command(resume={
    "event": "review_resolution_checked",
    "state_version": state_version,
})
```

`verify_review_resolution_from_postgres`가 현재 actor permission과 bound ReviewItem을
다시 읽는다.

- `/resume` preflight에서 하나라도 `pending_review`: `Command` 생성, graph invoke,
  saver write 없이 API `409 review_unresolved`
- 하나라도 `needs_more_evidence`: `needs_more_evidence`로 terminal
- 나머지가 모두 `approved`/`rejected`: `completed`

candidate set은 checkpoint 확인 뒤 immutable하다. 정상 API 경로에서 preflight를
통과한 뒤 pending item이 새로 생길 수 없다. repair나 내부 오용으로 graph node가
pending을 관찰하는 방어 경로에서는 `await_human_review`가 다시 interrupt하고, sync
saver confirmation 뒤 application status는 `awaiting_human_review`를 유지하며 API는
409를 반환한다. 이 경우에만 checkpoint id가 진행될 수 있고 business row는 바뀌지
않는다.

후보가 없는 initial run은 interrupt 없이 terminal이다. 후보가 있는 run은
`invoke_and_confirm_checkpoint(..., durability="sync")`가 returned/saved interrupt를
확인한 뒤에만 `awaiting_human_review`가 된다.

## 8. API 계약

### 8.1 Legacy

다음 endpoint와 field shape는 migration 기간에 유지한다.

- `GET /api/v1/orchestration/company-memory`
- `POST /api/v1/orchestration/company-memory/dry-run`
- `POST /api/v1/orchestration/company-memory/run`

실제 checkpoint가 아니므로 status 값은 사실대로 정정한다.

- `hitl_checkpointing=false`
- `checkpoint_store=none`
- `review_boundary=metadata_only`

V1은 간단해서 선택하는 새 경로가 아니라 backward compatibility와 rollback을 위한
임시 경로다. frontend는 한 실행에서 V1 또는 V2 중 하나만 호출한다.

### 8.2 V2 endpoint

- `GET /api/v1/orchestration/v2/company-memory`
- `POST /api/v1/orchestration/v2/company-memory/dry-run`
- `POST /api/v1/orchestration/v2/company-memory/runs`
- `GET /api/v1/orchestration/v2/company-memory/runs/{thread_id}`
- `POST /api/v1/orchestration/v2/company-memory/runs/{thread_id}/resume`
- `POST /api/v1/orchestration/v2/company-memory/runs/{thread_id}/cancel`

run/dry-run request는 다음 값만 허용한다.

```json
{
  "source_refs": [
    {
      "source_type": "gmail",
      "source_id": "message-opaque-id",
      "version_or_signature": "version-1"
    }
  ],
  "agent_names": [
    "mail_document_agent",
    "timeline_agent",
    "history_agent",
    "decision_record_agent",
    "todo_agent"
  ],
  "client_request_id": "optional-client-retry-key"
}
```

임의 question, objective, prompt, raw source content는 허용하지 않는다.

`agent_names`는 alias가 아니라 `AgentRegistry`에 등록된 실제 `AgentManifest.name`만
허용한다. Deliverable C의 non-Slack allowlist와 기본 순서는 위 다섯 이름이다.
`memory_extraction_agent` 같은 group alias는 public request에서 허용하지 않는다.
Integrations는 agent 선택 UI를 만들지 않고 V2 status가 제공하는
`default_agent_names`를 그대로 사용한다. `selection_policy_version`과 normalized
ordered names는 request HMAC, idempotency 비교와 effect key 계산에 포함한다.

preflight는 먼저 같은 owner의 `client_request_id` row를 조회한다. row가 있고 canonical
batch/graph/policy가 다르면 shared batch 존재 여부와 관계없이 409
`idempotency_key_reused`를 반환한다. exact match면 그 thread를 반환한다. 같은-owner
client row가 없을 때만 scope-level batch lock을 얻고, lock 안에서 client-id 검사를
한 번 더 반복한 뒤 shared batch를 조회 또는 생성한다. 따라서 다른 actor나 다른
client id의 exact batch는 기존 shared thread를 재사용하지만, client-id payload mismatch가
batch-first lookup으로 우회되지 않는다.

여기서 same-owner key는 `AgentWorkflowThread.owner_subject_id`가 현재 actor인, 즉 그
actor의 요청이 실제 thread를 생성하면서 저장된 key만 뜻한다. cross-owner shared
reuse는 existing thread를 반환할 뿐 caller별 alias나 key reservation을 만들지 않는다.
그러므로 public `client_request_id`는 모든 actor에 걸친 general idempotency key가 아니며,
지원 UI처럼 canonical job/policy에서 매번 결정적으로 생성하는 transport retry key로만
사용한다. 이 creator-only 의미를 바꾸려면 별도 alias persistence와 human gate가 필요해
이번 no-schema 범위에서는 제외한다.

dry-run은 같은 normalization, permission, evidence/version, agent allowlist와 cost
preflight를 사용하지만 thread, refs, checkpoint, AgentRun, ReviewItem을 쓰지 않는다.
Integrations는 dry-run에 사용한 정확한 source refs로 `/runs`를 호출한다. 그 사이
canonical version이 바뀌면 actual run은 `evidence_changed`로 중단한다.

후보가 있고 actual pause가 확인되면 HTTP 202를 반환한다.

```json
{
  "thread_id": "opaque-thread-id",
  "status": "awaiting_human_review",
  "review_item_count": 4,
  "review_status_counts": {"pending_review": 4},
  "durable": true,
  "graph_version": "company-memory-review-v2.0",
  "resume_allowed": false
}
```

후보가 없으면 terminal HTTP 200이다. `GET .../runs/{thread_id}`는
`thread_id`, `status`, `review_item_count`, `review_status_counts`, `durable`,
`graph_version`, `review_resolution_ready`, `checkpoint_resumable`, `resume_allowed`,
`retry_allowed`, `created_at`, `updated_at`, nullable allowlisted `error_code`와
`resume_error_code`를 반환한다. evidence, ReviewItem ids, checkpoint id, raw exception은
반환하지 않는다.

`resume`은 다음 두 lifecycle action을 같은 service로 처리한다.

- `awaiting_human_review`: DB resolution 확인 후 같은 graph thread resume
- `checkpoint_failed`: 기존 business rows를 보존한 checkpoint repair/reconciliation

다른 상태의 resume은 409다. `cancel`은 허용된 non-terminal 상태에서만 CAS로
전이하고 bound ReviewItem이나 knowledge를 삭제하지 않는다.

### 8.3 Feature flag와 readiness HTTP 계약

`GET /api/v1/orchestration/v2/company-memory`는 frontend가 CTA availability를 판단할
수 있도록 항상 인증된 200 diagnostic을 반환한다. 최소 field는 `enabled`,
`available`, `checkpoint_mode`, `durable`, `graph_version`,
`default_agent_names`, nullable allowlisted `error_code`다.

| 환경 | diagnostic | dry-run/new run | existing status/resume/cancel |
|---|---|---|---|
| flag off | `enabled=false`, `available=false`, mode `disabled` | 신규 dry-run/run은 404 | 등록된 builder가 있는 기존 thread의 status/resume/cancel은 유지 |
| flag on + SQLite/demo | available, mode `memory`, `durable=false` | 실행 가능 | process가 살아 있는 동안 가능; checkpoint 유실 시 resume 503 |
| flag on + ready PostgreSQL | available, mode `postgres`, `durable=true` | 실행 가능 | 실행 가능 |
| flag on + unready PostgreSQL | `available=false`, `checkpoint_unavailable` | 503, memory fallback 없음 | status는 200 진단, resume은 503, DB-only cancel은 허용 |

flag가 꺼져도 waiting V2 thread를 legacy graph로 보내지 않는다. 기존 status projection은
application DB에서 읽고, resume은 immutable `graph_version` builder와 원래 checkpoint
mode를 요구한다. unsupported builder는 non-mutating
`runtime_version_unavailable`이다.

### 8.4 Review API

기존 `GET /api/v1/review`에 optional `workflow_thread_id` filter를 추가한다. filter는
항상 기존 permission filtering 뒤에 적용한다. list endpoint에서 foreign/invisible
workflow 또는 visible item이 없는 workflow는 `items=[]`, `groups=[]`,
`total_count=0`을 반환해 존재 여부를 드러내지 않는다. 별도의 V2 thread status
endpoint는 foreign thread에 404를 반환한다.
Review page deep link는 다음 형식을 사용한다.

```text
/review?workflow_thread_id=<opaque-thread-id>
```

approval response에는 기존 field를 유지하면서 다음 stable metadata를 additive로
제공한다.

- `replayed`
- `promotion.target_type`
- `promotion.created_record_ids`
- `promotion.created_timeline_event_ids`

bulk response는 기존 approved/rejected/failed/skipped 결과를 유지하고 replay된 항목을
구분한다. Review API가 permission 확인 후 ReviewItem id를 반환하는 것은 허용되지만,
V2 workflow summary API는 이를 반환하지 않는다.

## 9. Authorization, permission, data minimization

- initial run, candidate persistence, Review action, resume에서 현재 permission을 각각
  재확인한다.
- role 이름에서 permission level을 추론하지 않고 actor의 정확한
  `allowed_permission_levels`를 사용한다.
- 여러 source가 있으면 strictest permission을 candidate와 output에 적용한다.
- agent는 visibility를 좁힐 수 있지만 넓힐 수 없다.
- foreign thread는 존재 여부를 숨기는 404다.
- 최초 owner는 모든 bound evidence에 대한 현재 visibility를 유지할 때 자신의
  thread를 status/resume/cancel할 수 있다.
- 같은 security scope의 다른 actor는 모든 bound evidence를 현재 볼 수 있을 때 shared
  thread status와 Review items를 볼 수 있다. non-owner 일반 사용자는 resume/cancel하지
  못하며, reviewer/admin은 모든 bound item에 대한 현재 권한이 있을 때 resume할 수
  있고 admin은 cancel할 수 있다.
- Review 승인 권한은 workflow owner 권한과 별도로 기존 Review RBAC가 검사한다.
- source id/URL/snippet, question, prompt, model output, provider error text는 checkpoint,
  V2 summary response, AuditLog에 넣지 않는다.
- integration audit의 기존 changed-source id 목록은 V2 경로에서
  `changed_source_count`와 keyed batch HMAC으로 대체한다. UI에 필요한 refs는
  authenticated sync/runtime response에서만 전달한다.
- audit은 actor/thread/graph version/outcome/error code와 bounded count만 allowlist로
  기록한다.

## 10. Transaction, failure recovery, concurrency

PostgreSQL business transaction과 LangGraph saver write를 하나의 분산 transaction으로
취급하지 않는다.

1. preflight transaction이 application thread, request, evidence refs를 생성 또는
   exact idempotent reuse한다.
2. draft worker가 row lock/CAS로 짧은 lease를 얻고 commit한다.
3. model/deterministic extraction은 transaction과 row lock 밖에서 실행한다.
4. persistence transaction이 lease, state version, cancellation, evidence signature,
   current permission을 다시 검사한다.
5. AgentRun과 ReviewItem을 unique key로 insert-or-return-existing하고
   `checkpoint_pending`으로 전이한다.
6. graph interrupt/checkpoint를 sync durability로 저장하고 exact tuple을 확인한다.
7. 별도 짧은 transaction이 `awaiting_human_review`로 전이한다.

실패별 계약은 다음과 같다.

| 실패 | 사용자 결과 | 보존/복구 |
|---|---|---|
| invalid/unauthorized source | 실행 불가 | business/checkpoint write 없음 |
| model 중 lease expiry | 같은 thread 내부 복구 | 계산 결과 폐기, 새 lease 전 candidate write 없음 |
| retry budget 소진 또는 명시적 cancel | 처리 실패 또는 취소됨 | terminal stable owner, user retry 없음 |
| draft commit 직후 crash | 재시도 필요 | thread keys로 AgentRun/ReviewItem reuse |
| checkpoint save/confirmation 실패 | 재시도 필요 | candidate 유지, `checkpoint_failed` |
| checkpoint 성공 후 app status write 실패 | 재시도 필요 | saved tuple로 status reconcile |
| unresolved review resume | 검토가 남아 있음 | 409, graph advance 없음 |
| concurrent resume/cancel | 다른 요청 처리 중 | 한 CAS만 성공, 나머지 409 |
| permission revoked | 권한 없음 | fail closed, 기존 knowledge 삭제 없음 |
| unsupported graph version | 현재 실행 재개 불가 | non-mutating `runtime_version_unavailable` |

repair는 같은 application thread와 idempotency keys를 사용한다. valid checkpoint가
있으면 application status만 reconcile한다. checkpoint가 없거나 손상되었으면 새 opaque
checkpoint-thread attempt를 발급하되 기존 AgentRun/ReviewItem을 재사용한다.

## 11. Cost 정책

- status, Review list, resume readiness는 paid LLM을 호출하지 않는다.
- dry-run은 deterministic preflight이며 provider invocation을 하지 않는다.
- `검토 후보 만들기` 클릭 전 estimated input/output token, cost, budget 상태를 표시한다.
- source window는 changed source refs 중 permission-filtered, ranked, deduped 범위로
  제한한다.
- evidence HMAC, prompt/model version, exact permission fingerprint로 cache를 재사용한다.
- cache hit와 unchanged source는 model call을 생략하지만 source evidence,
  permission, uncertainty를 제거하지 않는다.
- AgentRun에 estimated/actual token과 cost를 계속 기록한다.

## 12. 테스트 전략

모든 behavior change는 failing test를 먼저 확인하고 fake/deterministic model만 사용한다.

### 12.1 Contract와 unit

- source ref normalize/sort/dedupe와 version mismatch
- canonical source-type/prefixed-id/content-signature mapping
- async SyncJob의 `last_changed_sync_job_id` ref recovery
- 같은/different owner의 concurrent V2 launch가 scope-level shared thread 하나만 생성
- unauthorized cross-owner launch 404와 다른 security scope의 독립 thread
- same-owner client-id mismatch가 shared batch lookup보다 먼저 409
- cancelled/failed zero-effect thread를 포함한 exact batch stable reuse
- creator-only client-id binding과 cross-owner reuse의 unbound transport key
- `checkpoint_failed`만 retry 가능하고 failed/cancelled에는 retry action이 없음
- legacy historical marker 거부와 V2-mode sync waterline/rollback marker 전환
- V2 batch ownership과 legacy inline suppression
- HMAC fingerprint와 같은/different `client_request_id` reuse
- exact Registry manifest allowlist와 group alias 거부
- strictest permission과 foreign thread 404
- checkpoint-safe state와 sensitive value 거부
- 후보 없음 terminal, 후보 있음 actual interrupt
- exact interrupt payload와 같은 thread `Command(resume=...)`
- DB status가 resume acknowledgement보다 우선
- pending, needs-more-evidence, approved/rejected branch
- terminal Review transition 409
- single/bulk/concurrent promotion exactly once와 approval replay
- decision/history/todo companion Timeline provenance exactly once
- cancel, lease expiry, permission change, checkpoint repair

### 12.2 API

- flag off/memory/PostgreSQL ready/unready mode matrix
- actual pause만 202, no-candidate는 200
- 400/404/409/503 error mapping과 raw error 미노출
- V2 response에 source text/ref, item id, checkpoint id가 없음
- integration `changed_source_refs` additive compatibility
- legacy metadata-only status truthfulness
- Review workflow filter와 replay metadata
- live ReviewItem count projection과 `resume_allowed` 계산
- diagnostic/dry-run/run/status/resume/cancel mode별 HTTP matrix

### 12.3 PostgreSQL integration

- 실제 interrupt 뒤 app/pool을 닫고 독립 pool/saver로 같은 thread resume
- root checkpoint namespace와 immutable graph version
- 저장/복원된 tuple에 source content, URL, prompt, model output이 없음
- draft commit/checkpoint failure와 repair
- checkpoint 성공/status write 실패 reconciliation
- concurrent resume 한 건만 성공

### 12.4 Frontend/E2E

- Integrations에서 changed source 수와 cost 표시
- `검토 후보 만들기` 후 후보가 있으면 Review deep link
- Review workflow filter, progress, durability 안내
- 모든 item terminal 전 `검토 완료` disabled
- 승인/거절/추가 근거 후 explicit resume
- checkpoint retry와 needs-more-evidence 안내
- 기존 Review bulk/group/evidence drawer 동작 유지
- desktop/mobile production build와 affected Playwright

### 12.5 Regression gate

- Deliverable C focused backend suite
- Deliverable B runtime/checkpoint suite
- dedicated PostgreSQL restart suite
- 전체 non-Slack backend suite 신규 실패 0
- 전체 backend suite에서 현재 보류된 Slack 실패 열 건을 숨기지 않고 동일하게 보고
- touched backend Ruff `--no-fix`, `uv lock --check`, `git diff --check`
- frontend lint/build와 affected Playwright

## 13. Rollout과 rollback

### 13.1 Rollout

1. C implementation plan과 explicit coding authorization을 별도로 받는다.
2. 공통 Review transition/promotion service를 regression coverage와 함께 배포한다.
3. V2 route/UI는 `LANGGRAPH_REVIEW_V2_ENABLED=false`로 배포한다.
4. legacy API는 metadata-only checkpoint를 사실대로 표시한다.
5. SQLite/demo memory mode에서 deterministic smoke를 검증한다.
6. staging PostgreSQL에서 Gmail, Drive, Calendar와 fake/deterministic model로
   interrupt/restart/resume을 검증한다.
7. 준비된 staging/canary deployment에서 Integrations V2 CTA를 활성화한다.
8. cutover 이후 V2-mode sync가 표시한 source version만 CTA에 제공하고, 동일 batch의
   legacy inline candidate generation을 끈다.
9. 성공 기준 충족 후 V2를 기본 UI 경로로 전환한다.
10. V1 제거는 active thread drain과 별도 승인 후 다른 deliverable로 진행한다.

### 13.2 Rollback

- 신규 V2 run 생성 flag만 끈다.
- 이미 waiting인 V2 thread는 같은 graph builder로 status/resume/cancel을 유지한다.
- ReviewItem, approved knowledge, AgentRun, AuditLog를 삭제하거나 되돌리지 않는다.
- checkpoint schema를 자동 downgrade하지 않는다.
- UI는 신규 CTA를 숨기되 기존 Review Queue는 계속 사용한다.
- legacy V1은 신규 batch fallback으로 사용할 수 있지만 같은 batch를 V2와 중복
  실행하지 않는다.

## 14. 완료 기준

Deliverable C는 다음 조건을 모두 만족해야 완료다.

- 사용자가 Integrations와 Review 두 화면 안에서 후보 생성부터 완료까지 처리한다.
- 실제 LangGraph interrupt가 downstream을 멈추고 explicit resume이 같은 thread를
  이어간다.
- PostgreSQL ReviewItem과 current permission이 resolution source of truth다.
- retry, response loss, concurrency, crash가 ReviewItem, AgentRun, knowledge,
  companion Timeline 중복을 만들지 않는다.
- PostgreSQL app/pool restart 후 실제 resume이 성공한다.
- checkpoint, audit, V2 summary에 민감한 source/model data가 없다.
- V2가 disabled-by-default이고 production PostgreSQL 장애에서 memory로 강등되지 않는다.
- async sync 완료 뒤에도 canonical changed refs를 복구하고 동일 batch를 V1/V2가
  중복 처리하지 않는다.
- 같은 workspace의 권한 있는 여러 사용자가 exact batch에 ReviewItem/knowledge/Timeline
  중복을 만들지 않고 shared thread를 재사용한다.
- cutover 이전 legacy-processed source version은 V2 신규 candidate 대상이 아니다.
- legacy checkpoint 표현이 metadata-only로 정정된다.
- Review 기존 UX와 non-Slack regression에 신규 실패가 없다.
- Slack 열 건의 보류 실패는 그대로 보이고 C에서 수정하거나 숨기지 않는다.
- CDC/outbox/broker/streaming code가 추가되지 않는다.
- portfolio log와 session handoff에 실제 verification evidence가 기록된다.

## 15. 구현 경계와 다음 단계

이 문서 승인은 제품 코드 구현 승인이 아니다. 다음 단계는
`superpowers:writing-plans`로 별도의
`docs/superpowers/plans/2026-08-27-review-queue-hitl-v2.md` implementation plan을
작성하고 사용자 검토를 받는 것이다.

구현 plan은 최소한 다음 vertical slice로 나눈다.

1. legacy truthfulness, V2 schemas, canonical preflight
2. threaded draft와 actual interrupt
3. common locked Review transition/exactly-once promotion
4. status/resume/cancel/reconciliation
5. Integrations -> Review 최소 UX
6. PostgreSQL restart, regression, rollout documentation

2026-08-27 사용자는 exact canonical source-version batch의 workspace-shared ownership을
승인했다. 이 승인은 다른 source/version의 의미상 유사 timeline/history를 자동 병합하는
권한까지 포함하지 않는다.

output schema, permission policy, token budget, Review Queue trust boundary, trusted
knowledge promotion, duplicate timeline/history resolution에 이 문서와 다른 변경이
필요하면 구현을 멈추고 사용자 결정을 다시 받는다.
