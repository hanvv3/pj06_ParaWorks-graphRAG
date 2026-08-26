# LangChain·LangGraph Runtime Foundation 설계

## 1. 목적

ParaWorks의 GraphRAG 개발에 앞서 현재의 부분적인 LangChain·LangGraph 적용을
운영 가능한 런타임 구조로 고도화한다. 이 단계는 Neo4j를 붙이는 단계가 아니라,
이후 pgvector와 Neo4j GraphRAG를 동일한 오케스트레이션 경계 뒤에 연결할 수 있도록
실행·상태·분기·중단·재개 계약을 먼저 안정화하는 단계다.

이 설계의 완료 상태는 다음과 같다.

- 실제 LangGraph `StateGraph`가 typed state와 조건부 edge를 실행한다.
- Review Queue 경계가 일반 metadata가 아니라 `interrupt()`와 checkpointer를
  사용하는 실제 human-in-the-loop 실행 경계가 된다.
- 동일한 `thread_id`와 `Command(resume=...)`로 중단된 실행을 재개할 수 있다.
- production PostgreSQL과 test/smoke 환경이 동일한 checkpointer factory 계약을
  사용한다.
- 기존 `EvidencePacket`, `PermissionContext`, 비용·캐시·감사 계약은 유지한다.
- 이후 Neo4j retriever가 기존 pgvector/keyword 경로와 같은 포트로 추가될 수 있다.

## 2. 현재 상태와 문제

현재 저장소는 실제 라이브러리를 import하고 호출한다.

- `langchain.agents.create_agent`
- `ChatOpenAI`, `ChatGoogleGenerativeAI`
- `with_structured_output(...).invoke(...)`
- `langgraph.graph.StateGraph`
- graph `compile()`과 `invoke()`

그러나 현재 LangGraph graph는 전달받은 Python callable을 순서대로 실행하는 직선형
wrapper다. 조건부 routing, durable checkpoint, `interrupt()`,
`Command(resume=...)`, 안정적인 `thread_id`가 없다. 기존
`hitl_checkpoint` 값은 Review Queue 상태를 설명하는 dictionary이며 LangGraph가
저장한 checkpoint가 아니다.

RAG도 keyword/pgvector 검색, 권한 필터, evidence 조립, 모델 호출이 하나의 imperative
service 흐름에 묶여 있다. 이 상태에서 Neo4j를 먼저 추가하면 새로운 검색 로직을 현재
service에 넣은 뒤 다시 LangGraph node와 retriever 포트로 이동해야 한다.

## 3. 범위

### 포함

1. 2026-08-26 기준 안정 버전으로 LangChain 계열 의존성을 갱신하고 `uv.lock`을
   재생성한다.
2. typed LangGraph state와 작은 node 계약을 정의한다.
3. 실제 conditional routing을 사용하는 runtime graph를 만든다.
4. Review Queue 생성 후 실제 LangGraph interrupt를 발생시킨다.
5. test/smoke용 in-memory checkpointer와 production용 PostgreSQL checkpointer를
   factory 뒤에 둔다.
6. thread 생성, 중단 상태 조회, 검토 완료 후 resume API 계약을 추가한다.
7. keyword와 pgvector 검색을 공통 retriever 포트 뒤에 둔다.
8. permission/evidence/citation/cost trace 회귀 테스트를 추가한다.

### 제외

- Neo4j container, schema, constraint, vector index 생성
- `neo4j-graphrag` 설치와 retriever 구현
- graph projection/outbox 구현
- Text2Cypher
- Knowledge Map UI 변경
- 기존 pgvector 제거
- live LLM 또는 live embedding provider를 사용하는 테스트

Neo4j foundation과 GraphRAG retrieval은 이 런타임 기반이 green이 된 뒤 별도
spec과 plan으로 진행한다.

## 4. 의존성 정책

초기 upgrade target은 공식 PyPI에서 2026-08-26에 확인한 다음 안정 버전이다.

- `langchain>=1.3.17,<1.4.0`
- `langgraph>=1.2.11,<1.3.0`
- `langchain-openai>=1.6.0,<1.7.0`
- `langchain-google-genai>=4.3.5,<4.4.0`
- `langgraph-checkpoint-postgres>=3.1.2,<3.2.0`

`pyproject.toml`은 검증한 minor line을 제한하고, 실제 재현 버전은 `uv.lock`으로
고정한다. upgrade는 runtime 동작 변경과 별도 commit으로 분리한다. resolver가
호환 가능한 조합을 만들지 못하거나 기존 provider adapter 계약이 깨지면 임의의
compatibility shim을 만들지 않고 마지막 green 조합으로 범위를 낮춘 뒤 이유를
문서화한다.

## 5. 아키텍처

### 5.1 두 graph의 책임 분리

기존 `company_memory` 한 흐름이 지식 후보 생성과 질문 답변을 함께 처리하는 결합을
분리한다.

#### Company Memory Review Graph

```text
START
  -> validate_input
  -> collect_evidence
  -> plan_agent_runs
  -> draft_review_candidates
  -> route_review_boundary
       -> no_candidates -> END
       -> candidates_created -> await_human_review [interrupt]
  -> verify_review_resolution
       -> unresolved -> await_human_review
       -> resolved -> record_review_trace
  -> END
```

이 graph는 후보를 만드는 ingestion/review workflow다. AI 결과를 trusted knowledge로
직접 저장하지 않는다. 승인·반려·추가 근거 요청은 기존 Review Queue API가 수행한다.
resume node는 대상 `ReviewItem` 전체가 `approved` 또는 `rejected`인지 확인한 뒤 실행을
종료하고 trace만 기록한다. `needs_more_evidence`는 미해결 상태이므로 resume을 허용하지
않는다. knowledge promotion은 기존 Review Queue 승인 transaction을 계속 source of
truth로 사용한다.

#### RAG Answer Graph

```text
START
  -> validate_input
  -> resolve_permission_context
  -> classify_query
  -> select_retriever
       -> keyword
       -> pgvector
       -> neo4j        # 후속 단계에서 추가
  -> apply_permission_guard
  -> rank_evidence
  -> assemble_context
  -> generate_answer
  -> validate_citations
  -> record_cost_and_trace
  -> END
```

이번 단계에서는 keyword와 pgvector branch만 구현한다. Neo4j는 등록되지 않은 backend로
요청되면 조용히 keyword로 바꾸지 않고 명시적인 configuration error를 반환한다.
실행 중 pgvector 장애가 발생했을 때만 정책에 따라 keyword fallback을 허용하고
`fallback_used`, `error_category`를 trace에 남긴다.

### 5.2 파일 경계

- `backend/app/agent_runtime/state.py`
  - review graph와 RAG graph의 typed state를 정의한다.
- `backend/app/agent_runtime/checkpointing.py`
  - `InMemorySaver`와 `PostgresSaver` 선택, DSN 정규화, setup 책임을 가진다.
- `backend/app/agent_runtime/review_graph.py`
  - Review Queue interrupt/resume graph를 조립한다.
- `backend/app/agent_runtime/rag_graph.py`
  - retrieval routing과 answer graph를 조립한다.
- `backend/app/models/agent_workflow_thread.py`
  - graph `thread_id`, 소유 사용자, workflow 이름, 현재 상태와 timestamp를 저장한다.
- `backend/app/rag/retrievers/base.py`
  - retriever 요청·결과·trace 포트를 정의한다.
- `backend/app/rag/retrievers/keyword.py`
  - 현재 keyword retrieval을 포트에 맞춘다.
- `backend/app/rag/retrievers/pgvector.py`
  - 현재 pgvector adapter를 포트에 맞춘다.
- `backend/app/rag/retrievers/registry.py`
  - backend 선택을 담당하고 API route의 직접 import를 막는다.

기존 `backend/app/agent_runtime/contracts.py`의 공유 계약은 이동하거나 대규모로
바꾸지 않는다. 필요한 graph 전용 state는 별도 파일에 두어 source agent가
LangGraph 내부 표현에 결합되지 않게 한다.

## 6. 상태와 공개 계약

### 6.1 Review graph state

최소 state field는 다음과 같다.

- `thread_id: str`
- `objective: str`
- `question: str`
- `permission_context: PermissionContext`
- `evidence_packet_ids: list[str]`
- `cost_plan: dict[str, AgentCostPlan]`
- `review_item_ids: list[int]`
- `review_status: no_candidates | awaiting_human_review | resolved`
- `completed_nodes: list[str]`
- `errors: list[GraphError]`

checkpoint에는 API key, OAuth token, raw connector payload를 넣지 않는다. 근거는
stable ID와 검토에 필요한 제한된 metadata로 저장하고, 원문은 PostgreSQL에서 다시
읽는다.

### 6.2 RAG graph state

최소 state field는 다음과 같다.

- `question: str`
- `permission_context: PermissionContext`
- `query_type: fact | relation | impact | timeline`
- `requested_backend: keyword | pgvector | neo4j`
- `selected_backend: keyword | pgvector | neo4j`
- `retrieval_candidates: list[RetrievalCandidate]`
- `visible_candidates: list[RetrievalCandidate]`
- `hidden_match_count: int`
- `evidence_packet: EvidencePacket | None`
- `answer: RagAnswer | None`
- `trace: RetrievalTrace`
- `errors: list[GraphError]`

### 6.3 API 호환성

기존 status와 dry-run endpoint는 유지하되 실제 상태를 정확하게 표현한다.

- `GET /api/v1/orchestration/company-memory`
  - `checkpoint_store`를 설정에 따라 `memory` 또는 `postgres`로 반환한다.
- `POST /api/v1/orchestration/company-memory/dry-run`
  - DB write, interrupt, paid call 없이 graph topology와 route 결과만 보여준다.
- `POST /api/v1/orchestration/company-memory/run`
  - 새 `thread_id`를 생성하거나 요청에서 받은 id를 사용한다.
  - 후보가 있으면 `status=awaiting_human_review`와 interrupt payload를 반환한다.
  - 이 응답에서는 review 이후 node가 실행되지 않는다.
- `POST /api/v1/orchestration/company-memory/{thread_id}/resume`
  - 현재 사용자와 thread 소유권을 검증한다.
  - 대상 ReviewItem이 모두 `approved` 또는 `rejected`일 때만
    `Command(resume=...)`를 전달한다.
  - 미해결 항목이 있으면 409와 미해결 id 개수만 반환하고 원문을 노출하지 않는다.

기존 `/ask`, `/search`, assistant 응답의 citation·permission·cost shape은 유지한다.
내부 실행만 RAG Answer Graph로 이동한다.

## 7. Checkpoint와 transaction 정책

- test는 각 테스트마다 독립적인 `InMemorySaver`를 주입한다.
- SQLite smoke mode도 process-local `InMemorySaver`를 사용하며 재시작 복구를
  보장한다고 표시하지 않는다.
- PostgreSQL production mode는 `langgraph-checkpoint-postgres`의
  `PostgresSaver`를 사용한다.
- saver table setup은 application import 시 자동 실행하지 않는다. 명시적인
  bootstrap command 또는 deployment migration 단계에서 한 번 실행한다.
- graph `thread_id`와 사용자 id의 대응은 `AgentWorkflowThread` PostgreSQL record로
  검증한다. checkpoint metadata만으로 소유권을 판단하지 않는다.
- `draft_review_candidates` node가 ReviewItem 생성 transaction을 commit한 다음,
  side effect가 없는 별도 `await_human_review` node가 `interrupt()`를 호출한다.
- resume 전에 ReviewItem 상태를 PostgreSQL에서 다시 조회한다. checkpoint payload를
  trusted approval state로 간주하지 않는다.
- interrupt 전 side effect는 idempotent해야 한다. evidence hash와 기존 cache 계약으로
  동일 thread 재실행 시 ReviewItem과 AgentRun 중복 생성을 막는다.

## 8. LangChain 사용 원칙

LangChain의 `create_agent`는 모델이 tool을 선택하고 반복 실행해야 하는 실제 agentic
경로에만 사용한다. permission guard, approval 확인, backend selection처럼 결정적이고
보안에 민감한 단계는 LangGraph의 명시적인 node로 유지한다.

모델 출력은 가능한 경우 다음 공식 경계를 사용한다.

- `create_agent(..., response_format=PydanticSchema)`
- `model.with_structured_output(PydanticSchema)`
- LangChain tool decorator와 typed input schema

수동 JSON 문자열 파싱, 자체 tool loop, 자체 checkpoint engine, 자체 graph executor는
추가하지 않는다.

## 9. Permission·evidence guardrail

- retrieval 전에 `PermissionContext`를 확정한다.
- retriever query 자체에서 가능한 permission filter를 적용한다.
- graph의 `apply_permission_guard`에서 결과를 다시 검증한다.
- restricted evidence를 경유한 결과는 restricted보다 넓힐 수 없다.
- 권한 없는 candidate의 source id, snippet, relationship path는 answer context에 넣지
  않는다.
- hidden match는 count만 기록한다.
- citation은 `visible_candidates`에서 조립한 `EvidencePacket`의 source만 허용한다.
- citation validator가 존재하지 않는 source를 발견하면 answer를 성공 처리하지 않는다.
- evidence 없는 ReviewCandidate는 기존과 동일하게 거부한다.

## 10. 오류와 fallback

오류는 최소 다음 category로 분류한다.

- `invalid_input`
- `permission_denied`
- `retriever_not_configured`
- `retriever_unavailable`
- `checkpoint_unavailable`
- `review_unresolved`
- `model_unavailable`
- `citation_validation_failed`

PostgreSQL checkpointer가 production에서 초기화되지 않으면 in-memory로 자동 강등하지
않고 readiness error를 반환한다. pgvector가 선택된 실행 중 일시적으로 실패하면
keyword fallback을 사용할 수 있으나 반드시 trace와 API response에 표시한다.
permission 또는 citation 실패는 fallback으로 우회하지 않고 fail closed한다.

## 11. 테스트 전략

모든 behavior change는 실패 테스트를 먼저 확인한다.

### Dependency compatibility

- 모든 LangChain provider adapter import
- `create_agent` structured output contract
- `StateGraph` compile/invoke
- checkpoint package import

### Review graph

- 후보가 없으면 interrupt 없이 종료
- 후보가 있으면 `draft_review_candidates` 다음에 실제 interrupt
- interrupt 이후 node가 실행되지 않음
- 같은 `thread_id`와 `Command(resume=...)`로 재개
- 미해결 ReviewItem은 resume 거부
- resolved ReviewItem만 재개
- 다른 사용자는 thread를 재개할 수 없음
- 동일 thread retry가 ReviewItem/AgentRun을 중복 생성하지 않음

### RAG answer graph

- fact 질문은 설정된 vector/keyword branch로 routing
- relation/impact/timeline 분류가 typed state에 기록
- 미등록 Neo4j backend는 configuration error
- pgvector 장애 시 keyword fallback과 trace 기록
- permission filter가 생성 node 전에 적용
- restricted match의 원문과 path가 숨겨짐
- citation은 visible evidence에만 연결
- fake model output으로 전체 graph 실행

### Regression

- 기존 agent runtime contract tests
- RAG orchestrator tests
- Review Queue promotion tests
- orchestration API tests
- focused backend suite
- 전체 backend suite에서 기존 baseline 대비 신규 실패 0개

자동 테스트는 live LLM, live embedding, Slack, Gmail, Drive, OAuth API를 호출하지 않는다.

## 12. 단계적 전환과 rollback

1. 의존성 upgrade만 적용하고 기존 focused suite를 검증한다.
2. 새 checkpointer/retriever 포트와 graph를 기존 경로 옆에 추가한다.
3. dry-run과 테스트에서 새 graph를 먼저 사용한다.
4. company-memory run을 실제 interrupt graph로 전환한다.
5. `/ask`와 `/search` 내부를 RAG Answer Graph로 전환한다.
6. 동등성·권한·citation 검증 후 기존 linear wrapper를 제거한다.

전환 중에는 `LANGGRAPH_RUNTIME_V2_ENABLED` feature flag를 둔다. rollback은 flag를
끄고 기존 keyword/pgvector 서비스로 돌아가는 방식이다. 단, 새 graph에서 생성된
ReviewItem과 AgentRun은 PostgreSQL 공식 기록이므로 삭제하지 않는다. checkpoint는
파생 실행 상태이며 비활성화할 수 있다. flag가 꺼진 동안 status API는
`hitl_checkpointing=false`, `checkpoint_store=none`을 반환해 legacy metadata를 실제
checkpoint로 표현하지 않는다.

flag 제거 조건은 다음과 같다.

- focused runtime/RAG/review tests green
- 전체 backend baseline 대비 신규 실패 0개
- actual interrupt/resume smoke green
- permission leakage 0건
- citation contract 회귀 0건

## 13. 성공 기준

- 코드와 status API가 실제 checkpoint 상태를 과장하지 않는다.
- Review Queue 경계가 실제 LangGraph interrupt/resume으로 동작한다.
- production에서 PostgreSQL checkpointer를 사용하고 smoke/test에서는 주입 가능한
  in-memory saver를 사용한다.
- RAG 실행에 적어도 하나의 실제 conditional edge가 있다.
- keyword와 pgvector가 공통 retriever 포트로 실행된다.
- 이후 Neo4j retriever 추가가 API route나 source agent 수정 없이 가능하다.
- evidence, permission, review, citation, cache, cost 계약이 유지된다.
- dependency upgrade와 runtime behavior change가 분리된 commit으로 남는다.

## 14. 후속 단계

이 spec이 완료된 뒤 다음 순서로 별도 설계를 진행한다.

1. Neo4j foundation과 PostgreSQL outbox projection
2. approved knowledge graph sync
3. `neo4j-graphrag` VectorCypher/HybridCypher retriever
4. RAG Answer Graph의 `neo4j` branch 활성화
5. pgvector 비교 평가와 Knowledge Map 연결
