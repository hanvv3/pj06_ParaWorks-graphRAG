# E GraphRAG 구현 계획

**Goal:** 승인된 관계와 근거를 따라 검색하는 Neo4j adapter를 기존 RAG에 연결한다.
**Spec:** [E 설계](../specs/2026-09-20-e-graphrag-design.md).
**Architecture:** PostgreSQL-derived projection → bounded graph retrieval → canonical evidence 검증 → 기존 answer graph.
**Tech:** 기존 LangChain Runnable/LangGraph, PostgreSQL, 공식 Neo4j driver; 필요한 SDK만 착수 시 lock.
**상태:** 계획안; D.1 완료 뒤 E-1에서 fixture·relation/탐색 한도와 호환 버전을 확정한다.

## E-1 — 재실행 가능한 작은 projection

**파일 책임:** 신규 `backend/app/rag/graph_projection.py`(mapping/sync),
`graph_store.py`(driver 경계), 관련 admin job·모델/테스트.
기존 `indexing.py`, `serving_generation.py`, canonical models의 lifecycle를 재사용한다.

- [ ] 작은 관계 질문 fixture와 explicit relation allowlist, hop/candidate/time 한도를 확정한다.
  keyword/pgvector canonical 결과에서 traversal seed를 고르는 방식과 seed cap도 정한다.
  기존 pgvector 결과를 baseline으로 저장하고 첫 성공 사례를 정의한다.
- [ ] RED: 중복 sync·crash/restart·revoke/delete/supersede·scope 충돌·stale generation을 재현한다.
- [ ] stable identity/version 기반 bounded reconcile와 tombstone 처리·lag/count 관측을 구현한다.
  실제 Neo4j connection은 최소 권한, 자동 테스트는 fake driver로 시작한다.
- [ ] disposable Neo4j에서 재시작/중복/삭제 수렴을 검증하고 코드·증거를 커밋한다.

**Acceptance:** 동일 입력의 재실행이 중복 graph를 만들지 않고 PostgreSQL 현재 상태로 수렴한다.
Neo4j가 꺼져도 기존 RAG가 동작한다. outbox/CDC는 선행 조건이 아니다.

## E-2 — 권한을 보존하는 검색 adapter

**파일 책임:** 신규 `backend/app/rag/neo4j_retriever.py`, 기존 `retrieval.py`,
`backend/app/agent_runtime/rag_v2_contracts.py`, graph registry/routing와 D.1 key/dependency.
backend union 확장과 내부 version 변경의 소비자를 함께 점검한다.

- [ ] RED: relation 질의의 유효 provenance, restricted 중간 node/edge, scope 경계,
  stale/deleted path, graph 장애, bounded hidden count와 중복 embedding 거부를 검증한다.
- [ ] parameterized bounded traversal을 LangChain adapter로 구현하고 PostgreSQL canonical 근거를
  다시 확인한다. public citation이나 trusted status를 graph 응답에서 직접 만들지 않는다.
- [ ] path 영향 근거를 D.1 invalidation에 포함하고 fallback/disabled의 현재 RAG 계약을 보존한다.
- [ ] 캐시 on/off 모두 generation 도중 중간 edge revoke/version/permission 변경을 RED로 재현하고,
  전체 relation dependency를 최종 publication 검증에 포함한다. endpoint 문서 검증만 재사용하지 않는다.
- [ ] fake graph+실제 DB 계약, 기존 API/Assistant/캐시 회귀를 검증·리뷰하고 커밋한다.

**Acceptance:** 새 adapter가 동일 RetrievalResult를 제공하고 공개 route/화면을 늘리지 않는다.
지원하지 않는 질의와 권한/authority 오류를 graph text로 임의 보충하지 않는다.

## E-3 — 관계 질의 개선과 rollback 증명

- [ ] 같은 권한·corpus·질문 집합으로 pgvector/graph의 precision/recall, 관계 답변 성공,
  faithfulness, latency, 비용, projection lag를 비교한다.
- [ ] 품질·권한 기준을 먼저 만족하고 개선되는 질의군을 기록한다. baseline 없는 임의 SLO를 통과라 하지 않는다.
- [ ] shadow 후 제한 활성화 후보를 정하고 graph unavailable/stale/revoke/cache drift,
  flag rollback·PG/Neo4j 재시작을 검증한다.
- [ ] 코드 revision/fixture/결과를 한 evidence 항목으로 기록하고 운영 적용 여부를 제시한다.

**완료:** 관계 검색의 실제 이점, 현재 permission/evidence/citation 보존, 장애 복귀가 확인됨.
Text2Cypher·유료 classifier·Knowledge Map 확대·CDC는 별도 필요가 입증될 때 추가한다.
그 다음 작업은 Slack 데이터 선택과 복구 계획이며, 실제 source가 없으면 synthetic 계약 검증부터 진행한다.
